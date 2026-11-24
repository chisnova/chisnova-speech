package com.chisnova.mercury.asr;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.ArrayList;

import org.json.JSONObject;
import org.json.JSONArray;
import com.chisnova.mercury.audio.WavFile;
import com.chisnova.mercury.itf.DecoderCallback;
import com.chisnova.mercury.jni.AsrAPI;

public class ApiTest {
    private static String decode(AsrAPI api, DecoderCallback callback, Path path) throws Exception {
        System.out.println("WAV: " + path);
        try (InputStream stream = Files.newInputStream(path)) {
            WavFile wav = WavFile.openWavFile(stream);
            if (wav.getNumChannels() != 1 || wav.getSampleRate() != 16000 ||
                    wav.getValidBits() != 16 || wav.getNumFrames() == 0) {
                throw new IOException("Expected a nonempty 16 kHz mono 16-bit PCM WAV: " + path);
            }
            api.clear();
            api.setCallBack(callback);
            api.startBackGroundEngine(3);
            double[] buffer = new double[160];
            float[] audio = new float[buffer.length];
            long remaining = wav.getNumFrames();
            while (remaining > 0) {
                int frames = wav.readFrames(buffer, (int)Math.min(remaining, buffer.length));
                if (frames == 0) throw new IOException("Truncated WAV: " + path);
                for (int j = 0; j < frames; j++) audio[j] = (float)buffer[j];
                remaining -= frames;
                api.sendAudio(audio, frames, remaining == 0);
                Thread.sleep(10);
            }
        }
        api.finishDecoding();
        System.out.println("CTC ASR RESULT: " + api.getCtcAsrResult());
        String result = api.getSecondPassDecodingResult(3);
        System.out.println("SECOND PASS DECODING RESULT: " + result);
        return result;
    }

    private static void score(AsrAPI asrApi, String jsonLine, String secondPassDecodingResult) {
        JSONObject jsonObject = new JSONObject(jsonLine);

        JSONArray org_words_array = jsonObject.getJSONArray("org");
        JSONArray word_idx_array = jsonObject.getJSONArray("word_idx");
        JSONArray word_idx_group_array = jsonObject.getJSONArray("word_idx_group");
        JSONArray phn_idx_array = jsonObject.getJSONArray("phn_idx");
        JSONArray phn_idx_group_array = jsonObject.getJSONArray("phn_idx_group");

        List<String> org_words = new ArrayList<String>();
        int[] word_idx = new int[word_idx_array.length()];
        int[] word_idx_group = new int[word_idx_group_array.length()];
        int[] phn_idx = new int[phn_idx_array.length()];
        int[] phn_idx_group = new int[phn_idx_group_array.length()];

        for (int i = 0; i < word_idx_array.length(); i++) {
            word_idx[i] = word_idx_array.getInt(i);
        }
        for (int i = 0; i < word_idx_group_array.length(); i++) {
            word_idx_group[i] = word_idx_group_array.getInt(i);
        }
        for (int i = 0; i < phn_idx_array.length(); i++) {
            phn_idx[i] = phn_idx_array.getInt(i);
        }
        for (int i = 0; i < phn_idx_group_array.length(); i++) {
            phn_idx_group[i] = phn_idx_group_array.getInt(i);
        }

        for (int i = 0; i < org_words_array.length(); i++) {
            org_words.add(org_words_array.getString(i));
        }

        String org_words_str = "";
        for (int i = 0; i < org_words.size(); i++) {
            org_words_str += org_words.get(i) + " ";
        }
        System.out.println("org_words: " + org_words_str);

        String hyp = secondPassDecodingResult.trim();
        String ref = org_words_str.trim();
        boolean match = ref.equalsIgnoreCase(hyp);
        System.out.println("LABEL_COMPARE: ref=\"" + ref + "\" | hyp=\"" + hyp + "\" | match=" + match);

        String pronounceScore = asrApi.getPronounceScore(org_words_str, word_idx, word_idx_group, word_idx.length, phn_idx, phn_idx_group, phn_idx.length);
        System.out.println("PRONOUNCE SCORE: " + pronounceScore);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1 && args.length != 2) {
            throw new IllegalArgumentException("Usage: java_api_test MODEL_DIR WAV_LIST or java_api_test SAMPLE_COUNT");
        }
        boolean fixture = args.length == 1;
        Path data = Paths.get(System.getProperty("mercury.test.data", "."));
        Path model = fixture ? Paths.get(System.getenv().getOrDefault("MERCURY_TEST_MODEL_DIR",
                System.getProperty("mercury.test.model", "python/binary"))) : Paths.get(args[0]);
        Path list = (fixture ? data.resolve("wav.list") : Paths.get(args[1])).toAbsolutePath();
        if (!Files.isRegularFile(model.resolve("model.json"))) throw new IOException("Missing model.json in " + model);
        int limit = fixture ? Integer.parseInt(args[0]) : Integer.MAX_VALUE;
        if (limit < 0) throw new IllegalArgumentException("Sample count must be nonnegative");
        System.loadLibrary("asr_api");
        try (BufferedReader reader = Files.newBufferedReader(list);
             BufferedReader labels = fixture ? Files.newBufferedReader(data.resolve("conan.jsonl")) : null) {
            AsrAPI api = new AsrAPI(model.toString());
            DecoderCallback callback = result -> System.out.println(result);
            try {
                int count = 0;
                String line;
                while (count < limit && (line = reader.readLine()) != null) {
                    String entry = line.trim();
                    if (entry.isEmpty()) continue;
                    Path path = list.getParent().resolve(entry);
                    String transcript = decode(api, callback, path);
                    if (fixture) {
                        String label = labels.readLine();
                        if (label == null) throw new IOException("Missing label for " + path);
                        score(api, label, transcript);
                    }
                    api.cancelDecoding();
                    count++;
                }
                if (limit > 0 && count == 0) throw new IOException("Empty WAV list: " + list);
            } finally {
                api.delete();
            }
        }
    }
}
