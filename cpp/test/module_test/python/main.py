"""Generate PyTorch reference tensors and C++ manifests for module_tester."""
import json
import math
from pathlib import Path

import torch
from torch import nn

from exporter.model_saver import save_model
from layers.conformer_conv_module import ConformerConvModule
from layers.conformer_conv_subsample import ConformerConvSubSample
from layers.conformer_feedforward import ConformerFeedForward
from layers.conformer_multihead import ConformerAttention, ConformerMultiHead
from layers.conformer_positional import ConformerPositionalEncoding
from layers.conformer_stream_conv_module import ConformerStreamConvModule
from layers.conformer_stream_multihead import ConformerStreamAttention, ConformerStreamMultiHead
from layers.feedforward import FeedForward
from layers.multihead import MultiHead
from layers.positional import PositionalEncoding
from module.conformer_encoder import ConformerEncoder, ConformerEncoderLayer
from module.conformer_stream_encoder import ConformerStreamEncoderLayer
from module.transformer_decoder import TransformerDecoder, TransformerDecoderLayer


DIM, HEADS, HIDDEN, VOCAB = 32, 4, 64, 48
BINARY = Path("binary")


def tensor_file(name, tensor):
    # The native tester reads row-major matrices. Token inputs have shape (1, U).
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    if tensor.dtype == torch.long:
        tensor = tensor.int()
    filename = name + ".pt"
    tensor.contiguous().numpy().tofile(BINARY / filename)
    return {"value": filename, "shape": list(tensor.shape)}


class LegacyEncoderLayer(nn.Module):
    """Post-norm Transformer retained by the native encoder."""
    def __init__(self):
        super().__init__()
        self.multihead = MultiHead(DIM, HEADS, False)
        self.ffn = FeedForward(DIM, HIDDEN)
        self.layer_norm1 = nn.LayerNorm(DIM)
        self.layer_norm2 = nn.LayerNorm(DIM)

    def forward(self, x):
        x = self.layer_norm1(x + self.multihead(x, x, x))
        return self.layer_norm2(x + self.ffn(x))


class PredictorLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.multihead1 = MultiHead(DIM, HEADS, True)
        self.ffn = FeedForward(DIM, HIDDEN)
        self.layer_norm1 = nn.LayerNorm(DIM)
        self.layer_norm2 = nn.LayerNorm(DIM)

    def forward(self, x):
        norm = self.layer_norm1(x)
        x = x + self.multihead1(norm, norm, norm)
        return x + self.ffn(self.layer_norm2(x))


class LegacyTransducer(nn.Module):
    """Single-token prediction and joint network for the legacy C++ runtime."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos_enc = PositionalEncoding(DIM)
        self.layers = nn.ModuleList([PredictorLayer()])
        self.enc_proj = nn.Linear(DIM, DIM)
        self.prd_proj = nn.Linear(DIM, DIM)
        self.fc1 = nn.Linear(2 * DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, VOCAB)

    def forward(self, tokens, enc):
        pred = self.pos_enc(self.emb(tokens))
        for layer in self.layers:
            pred = layer(pred)
        joint = torch.cat((self.enc_proj(enc), self.prd_proj(pred).expand_as(enc)), dim=-1)
        return self.fc2(torch.tanh(self.fc1(joint)))


@torch.no_grad()
def main():
    BINARY.mkdir(exist_ok=True)
    torch.manual_seed(42)
    cases = {}
    x = torch.randn(1, 23, DIM)
    acoustic = torch.randn(1, 51, 80)
    tokens = torch.randint(VOCAB, (1, 7))
    token = tokens[:, :1]

    def weights(name, model):
        return save_model(model.eval(), name, str(BINARY))

    def record(name, manifest, output, **inputs):
        case = {key: tensor_file(name + "_" + key, value) for key, value in inputs.items()}
        case["output"] = tensor_file(name + "_output", output)
        case["model"] = manifest
        cases[name] = case

    def single(name, module, *args, stream=False):
        manifest = weights(name, module)
        out = module.stream(*args) if stream else module(*args)
        record(name, manifest, out, input=args[0])
        return manifest

    single("conformer_conv_module", ConformerConvModule(DIM), x)
    sub = ConformerConvSubSample(1, 8)
    sub_manifest = single("conformer_conv_subsample", sub, acoustic)
    single("conformer_feedforward", ConformerFeedForward(DIM, 4, 0), x)
    rel = ConformerMultiHead(DIM, HEADS)
    single("conformer_multihead", rel, x, x, x, x)
    single("conformer_attention", ConformerAttention(DIM, HEADS), x, x)
    single("transformer_feedforward", FeedForward(DIM, HIDDEN), x)
    single("transformer_multihead", MultiHead(DIM, HEADS, True), x, x, x)

    # Individual heads now consume projected tensors. Their parent owns QKV weights.
    qk = x @ x.transpose(-1, -2)
    relative = rel.relative_shift(qk.unsqueeze(1)).squeeze(1)
    record("conformer_head", {}, ((qk + relative) / math.sqrt(DIM)).softmax(-1) @ x, input=x)
    causal = torch.ones_like(qk, dtype=torch.bool).triu(1)
    scaled = qk / math.sqrt(DIM)
    record("transformer_head", {}, scaled.masked_fill(causal, -torch.inf).softmax(-1) @ x, input=x)
    record("conformer_stream_head", {}, scaled.softmax(-1) @ x, input=x)
    single("conformer_stream_multihead", ConformerStreamMultiHead(DIM, HEADS), x, x, x, stream=True)
    single("conformer_stream_attention", ConformerStreamAttention(DIM, HEADS), x, stream=True)
    single("conformer_stream_conv_module", ConformerStreamConvModule(DIM, ksize=5), x, stream=True)
    stream_layer = ConformerStreamEncoderLayer(DIM, HEADS, 5, 0)
    stream_manifest = single("conformer_stream_encoder_layer", stream_layer, x, stream=True)
    enc_layer = ConformerEncoderLayer(DIM, HEADS, 0)
    enc_manifest = single("conformer_encoder_layer", enc_layer, x, x)

    pos = PositionalEncoding(DIM)
    pos_manifest = weights("position", pos)
    encoder = LegacyEncoderLayer()
    layer_manifest = single("transformer_encoder_layer", encoder, x)
    record("transformer_encoder", dict(num_layer=1, pos_enc=pos_manifest, layers={"0": layer_manifest}),
           encoder(pos(x)), input=x)
    decoder = TransformerDecoderLayer(DIM, HIDDEN, HEADS, 0)
    layer_manifest = single("transformer_decoder_layer", decoder, x, x)
    record("transformer_decoder", dict(num_layer=1, pos_enc=pos_manifest, layers={"0": layer_manifest}),
           decoder(pos(x), x), input=x)
    word = TransformerDecoder(DIM, HIDDEN, HEADS, 1, VOCAB, 0)
    word_manifest = weights("word_decoder", word)
    record("conformer_decoder", word_manifest, word(tokens, x), enc_input=x, dec_input=tokens)
    cascade = ConformerEncoder(DIM, HEADS, 1, 0)
    ctc_manifest = single("conformer_ctc", cascade, x)

    project = nn.Linear(sub(acoustic).size(-1), DIM)
    features = project(sub(acoustic))
    relative_pos = ConformerPositionalEncoding(DIM)
    base = dict(num_layer=1, num_head=HEADS, dim_model=DIM, feat_convsubsample=sub_manifest,
                feat_linear=weights("feat_linear", project))
    enc = dict(base, pos_enc=weights("relative_pos", relative_pos), layers={"0": enc_manifest})
    record("conformer_encoder", enc, enc_layer(features, relative_pos(features)), input=acoustic)
    # V2 used additive positions before the deployed encoder switched to RoPE.
    feat_norm, norm = nn.LayerNorm(80), nn.LayerNorm(DIM)
    v2 = dict(base, layers={"0": stream_manifest}, pos_enc=pos_manifest,
              feat_norm=weights("feat_norm", feat_norm), layer_norm1=weights("norm", norm))
    stream_layer.clear()
    encoded = stream_layer.stream(pos(norm(project(sub(feat_norm(acoustic))))))
    record("conformer_stream_encoderV2", v2, encoded, input=acoustic)

    single("transformer_transducer_layer", PredictorLayer(), x)
    transducer = LegacyTransducer()
    predictor_manifest = dict(weights("predictor", transducer), num_layer=len(transducer.layers))
    record("transformer_transducer", predictor_manifest, transducer(token, x),
           enc_input=x, dec_input=token)
    integration = dict(stream_encoder=v2, decoder=predictor_manifest, ctc_encoder=ctc_manifest,
                       wrd_decoder=word_manifest, phn_decoder=word_manifest,
                       ctc_linear1=weights("ctc1", nn.Linear(DIM, VOCAB)),
                       ctc_linear2=weights("ctc2", nn.Linear(DIM, VOCAB)))
    record("asr_stream_conformer_trf_transducer", integration, transducer(token, encoded),
           enc_input=acoustic, dec_input=token)
    manifest = BINARY / "module_test.json"
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps({"module_test": cases}, indent=2))
    temporary.replace(manifest)


if __name__ == "__main__":
    main()
