package com.chisnova.mercury.jni;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import androidx.test.core.app.ApplicationProvider;

import org.junit.Test;
import com.chisnova.mercury.itf.DecoderCallback;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class AsrDecoderTest { // extends TestCase {
    public void copyFile(String srcFile, AssetManager assetMgr, Context testContext) {
//        AssetManager assetMgr = this.getAssets();

        InputStream is = null;
        OutputStream os = null;
        try {
            String destFile = testContext.getFilesDir() + File.separator + srcFile;
//            Log.d("[AsrDecoder] file copy start", destFile);
            is = assetMgr.open(srcFile);
            os = Files.newOutputStream(Paths.get(destFile));

            byte[] buffer = new byte[1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            is.close();
            os.flush();
            os.close();
//            Log.d("[AsrDecoder] file copy done", destFile);

        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void copyAssetAll(String srcPath, AssetManager assetMgr, Context testContext) {
        String[] assets = null;
        try {
            assets = assetMgr.list(srcPath);
            assert assets != null;
            if (assets.length == 0) {
                copyFile(srcPath, assetMgr, testContext);
            } else {
                String destPath = testContext.getFilesDir() + File.separator + srcPath;
//                Log.d("[AsrDecoder] copyAssetAll", destPath);
                File dir = new File(destPath);
                if (!dir.exists()) {
                    dir.mkdir();
//                    Log.d("[AsrDecoder] copyAssetAll mkdir: ", destPath);
                }
                for (String element : assets) {
                    copyAssetAll(srcPath + File.separator + element, assetMgr, testContext);
                }
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void DecodingTest() throws IOException {
        // Chunk size as used in ApiTest
        final int chunkSize = 160;
        float[] audio = new float[chunkSize];
        Context testContext = ApplicationProvider.getApplicationContext();
        AssetManager asset_manager = testContext.getAssets();
        // model path

        String[] labels = {"i am honored to be with you today at your commencement from one of the finest universities in the world", "truth be told I never graduated from college and ah this is the closest I've ever gotten to a college graduation", "today i want to tell you three stories ", "that's it no big deal just three stories"};    

        copyAssetAll("model_state", asset_manager, testContext);
        String model_path = testContext.getFilesDir() + File.separator + "model_state";
        Log.i("[AsrAPI]", "Engine Init Start");
        AsrAPI asr_api = new AsrAPI(model_path);
        Log.i("[AsrAPI]", "Engine Init Done");

        // Optional: register callback similar to ApiTest for intermediate results
        asr_api.setCallBack(new DecoderCallback() {
            @Override
            public void onResult(String asrResult) {
                Log.i("[AsrAPI]", "Callback ASR RESULT: " + asrResult);
            }
        });
        asr_api.clear();
        asr_api.startBackGroundEngine(3);

        Log.i("[AsrAPI]","load proceessing folder start");
        AssetManager audioAssets = androidx.test.platform.app.InstrumentationRegistry.getInstrumentation().getContext().getAssets();
        Log.d("[AsrAPI]", Arrays.toString(audioAssets.list("google_steve")));
        String[] audio_list = audioAssets.list("google_steve");
        Log.i("[AsrAPI]","load proceessing folder done");

        for(int i=0; i<1; ++i) {
            try {
                assert audio_list != null;
                String audio_path = "google_steve" + File.separator + audio_list[i];
                Log.i("[AsrAPI]", audio_path);
                InputStream audio_stream = audioAssets.open(audio_path);
                WavFile wavFile = WavFile.openWavFile(audio_stream);
                wavFile.display();
                int numChannels = wavFile.getNumChannels();

                // Stream audio in 160-frame chunks like ApiTest
                double[] buffer = new double[chunkSize * numChannels];
                int framesRead;
                int audioSize = (int) wavFile.getNumFrames() * numChannels;
                int numChunk = audioSize / chunkSize;
                int remainChunk = audioSize % chunkSize;

                Log.i("[AsrAPI]", "Decoding start");


                for (int chunkIdx = 0; chunkIdx < numChunk; chunkIdx++) {
                    framesRead = wavFile.readFrames(buffer, chunkSize);
                    if (framesRead == 0) break;
                    for (int j = 0; j < framesRead; j++) {
                        audio[j] = (float) buffer[j];
                    }
                    asr_api.sendAudio(audio, framesRead, false);
                    try { Thread.sleep(10); } catch (InterruptedException ignored) {}
                }

                if (remainChunk > 0) {
                    framesRead = wavFile.readFrames(buffer, remainChunk);
                    for (int j = 0; j < framesRead; j++) {
                        audio[j] = (float) buffer[j];
                    }
                    Log.i("[AsrAPI]", "send remain final audio " + framesRead);
                    asr_api.sendAudio(audio, framesRead, true);
                    try { Thread.sleep(10); } catch (InterruptedException ignored) {}
                } else {
                    Log.i("[AsrAPI]", "send final audio 0");
                    asr_api.sendAudio(audio, 0, true);
                }

                asr_api.finishDecoding();
                Log.i("[AsrAPI]", "finish decoding");

                String ctcResult = asr_api.getCtcAsrResult();
                Log.i("[AsrAPI]", "CTC ASR RESULT: " + ctcResult);
                String secondPass = asr_api.getSecondPassDecodingResult(3);
                Log.i("[AsrAPI]", "SECOND PASS DECODING RESULT: " + secondPass);
                // 발음 점수는 라벨만으로 부를 수 없다 — 라벨만 받던 오버로드는 네이티브 구현이
                // 없어(같은 이름의 7-arg 함수에 바인딩되어 SIGSEGV) 제거했다. 채점 인자를 갖춘
                // 시험은 AsrDecodingRegressionTest / AsrRobustnessTest 에 있다.

                asr_api.cancelDecoding();
                break;
            } catch (WavFileException | IOException e) {
                Log.i("[AsrAPI]", "Something is wrong");
                e.printStackTrace();
            }
        }
    }

}
