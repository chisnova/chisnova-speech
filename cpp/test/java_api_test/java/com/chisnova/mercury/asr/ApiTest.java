package com.chisnova.mercury.asr;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.FileReader;
import java.io.IOException;

import org.json.JSONObject;
import org.json.JSONArray;

import com.chisnova.mercury.audio.WavFile;
import com.chisnova.mercury.itf.DecoderCallback;
import com.chisnova.mercury.jni.AsrAPI;

import java.util.List;
import java.util.ArrayList;

class ApiTestCallback implements DecoderCallback {
    private volatile String lastResult = "";

    @Override
    public void onResult(String asrResult) {
        System.out.println(asrResult);
        lastResult = asrResult;
    }

    public String getLastResult() {
        return lastResult;
    }

    public void clearLastResult() {
        lastResult = "";
    }
}

public class ApiTest {
    public static void main(String[] args) throws Exception, IOException {
        int maxCount = 5;
        if (args.length > 0) {
            try {
                maxCount = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.out.println("Invalid max_count argument, using default: " + maxCount);
            }
        }
        System.out.println("MAX_TEST_COUNT: " + maxCount);

        System.loadLibrary("asr_api");
        String assets = System.getenv().getOrDefault("MERCURY_TEST_MODEL_DIR", "python/binary");
        AsrAPI asrApi = new AsrAPI(assets);
        float[] audio = new float[160];

        String path;
        String label;
        BufferedReader reader = new BufferedReader(new FileReader("wav.list"));
        // BufferedReader labelReader = new BufferedReader(new FileReader("label.list"));
        BufferedReader jsonlReader = new BufferedReader(new FileReader("conan.jsonl"));
        java.util.List<String> jsonlLines = new java.util.ArrayList<>();
        String jsonlLine;
        while ((jsonlLine = jsonlReader.readLine()) != null) {
            jsonlLines.add(jsonlLine);
        }
        jsonlReader.close();

        ApiTestCallback callback = new ApiTestCallback();

        int count = 0;
        while ((path = reader.readLine()) != null && count < maxCount) {
            System.out.println(path);
            String json_obj = jsonlLines.get(count);
            count++;
            JSONObject jsonObject = new JSONObject(json_obj);

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
            
            asrApi.clear();
            asrApi.setCallBack(callback);
            asrApi.startBackGroundEngine(3);

            InputStream audio_stream = new FileInputStream(path);
            WavFile wavFile = WavFile.openWavFile(audio_stream);
            // wavFile.display();
            int numChannels = wavFile.getNumChannels();

            double[] buffer = new double[160 * numChannels];
            int framesRead;

            int chunk_size = 160;
            int chunk_idx = 0;
            int audio_size = (int)wavFile.getNumFrames() * numChannels;
            int num_chunk = (int)(audio_size / chunk_size);
            int remain_chunk = audio_size % chunk_size;
            for (chunk_idx = 0; chunk_idx < num_chunk; chunk_idx ++) {
                framesRead = wavFile.readFrames(buffer, chunk_size);
                if (framesRead == 0) {
                    break;
                }
                for (int j = 0; j < framesRead; j++) {
                    audio[j] = (float)buffer[j];
                }
                asrApi.sendAudio(audio, framesRead, false);
                Thread.sleep(10);
            }
            if (remain_chunk > 0) {
                framesRead = wavFile.readFrames(buffer, remain_chunk);
                for (int j = 0; j < framesRead; j++) {
                    audio[j] = (float)buffer[j];
                }
                System.out.println("send remain final audio " + framesRead);
                asrApi.sendAudio(audio, framesRead, true);
                Thread.sleep(10);
            }
            else{
                System.out.println("send final audio " + 0);
                asrApi.sendAudio(audio, 0, true);
            }
            asrApi.finishDecoding();
            System.out.println("finish decoding");
            // stream_conformer(2-pass cascade)의 최종 결과는 CTC/2패스에서 나온다 —
            // 스트리밍 콜백(callback)은 partial 용도라 라벨 비교에 쓰지 않는다
            String ctcResult = asrApi.getCtcAsrResult();
            System.out.println("CTC ASR RESULT: " + ctcResult);
            String secondPassDecodingResult = asrApi.getSecondPassDecodingResult(3);
            System.out.println("SECOND PASS DECODING RESULT: " + secondPassDecodingResult);

            // Label comparison: compare second pass result with reference
            String hyp = secondPassDecodingResult.trim();
            String ref = org_words_str.trim();
            boolean match = ref.equalsIgnoreCase(hyp);
            System.out.println("LABEL_COMPARE: ref=\"" + ref + "\" | hyp=\"" + hyp + "\" | match=" + match);
            callback.clearLastResult();

            String pronounceScore = asrApi.getPronounceScore(org_words_str, word_idx, word_idx_group, word_idx.length, phn_idx, phn_idx_group, phn_idx.length);
            System.out.println("PRONOUNCE SCORE: " + pronounceScore);
            asrApi.cancelDecoding();
            // break;
        }
        
        System.out.println("finish main thread");
    }
}
