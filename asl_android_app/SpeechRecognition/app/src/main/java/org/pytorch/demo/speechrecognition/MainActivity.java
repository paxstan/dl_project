package org.pytorch.demo.speechrecognition;

import android.content.Context;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Locale;

import org.pytorch.LiteModuleLoader;
import org.pytorch.demo.speechrecognition.databinding.ActivityMainBinding;


public class MainActivity extends AppCompatActivity implements Runnable {
    private static final String TAG = MainActivity.class.getName();
    ActivityMainBinding binding;
    private Module module;
    private TextView mTextView;

    private Button mButton;
    GridView gridView;
    ImageView imageView;
    private static final long START_TIME_IN_MILLIS = 60000;
    private long mTimeLeftInMillis = START_TIME_IN_MILLIS;
    private final static int REQUEST_RECORD_AUDIO = 13;
    private final static int AUDIO_LEN_IN_SECOND = 4;
    private final static int SAMPLE_RATE = 16000;
    public int RECORDING_LENGTH = SAMPLE_RATE*AUDIO_LEN_IN_SECOND ;

    private final static String LOG_TAG = MainActivity.class.getSimpleName();

    private int mStart = 1;
    private HandlerThread mTimerThread;
    private Handler mTimerHandler;
    private Runnable mRunnable = new Runnable() {
        @Override
        public void run() {
            mTimerHandler.postDelayed(mRunnable, 1000);

            MainActivity.this.runOnUiThread(
                    () -> {
                        mButton.setText(String.format("Listening - %ds left", AUDIO_LEN_IN_SECOND - mStart));
                        mStart += 1;
                    });
        }
    };

    @Override
    protected void onDestroy() {
        //stopTimerThread();
        super.onDestroy();
    }

    protected void stopTimerThread() {
        mTimerThread.quitSafely();
        try {
            mTimerThread.join();
            mTimerThread = null;
            mTimerHandler = null;
            mStart = 1;
        } catch (InterruptedException e) {
            Log.e(TAG, "Error on stopping background thread", e);
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        //setContentView(R.layout.activity_main);
        gridView = findViewById(R.id.grid_View);
        mButton = findViewById(R.id.btnRecognize);
        mButton.setEnabled(true);
        mTextView = findViewById(R.id.tvResult);
        imageView = findViewById(R.id.image);

        mButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButton.setText(String.format("Listening - %ds left", AUDIO_LEN_IN_SECOND - mStart));
                mButton.setEnabled(false);
                Thread thread = new Thread(MainActivity.this);
                thread.start();


                mTimerThread = new HandlerThread("Timer");
                mTimerThread.start();
                mTimerHandler = new Handler(mTimerThread.getLooper());
                mTimerHandler.postDelayed(mRunnable, 1000);

            }
        });
    }
    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[]{android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    private String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            Log.d("STATE", file.getAbsolutePath());
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, assetName + ": " + e.getLocalizedMessage());
        }
        return null;
    }

    private void showTranslationResult(String result) {
        mTextView.setText(result);

    }

    public void run() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        AudioRecord record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT, SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(LOG_TAG, "Audio Record can't initialize!");
            return;
        }
        record.startRecording();
        long shortsRead = 0;
        int recordingOffset = 0;
        short[] audioBuffer = new short[bufferSize / 2];
        short[] recordingBuffer = new short[RECORDING_LENGTH];



        while (shortsRead < RECORDING_LENGTH) {
            int numberOfShort = record.read(audioBuffer, 0, audioBuffer.length);
            shortsRead += numberOfShort;
            System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberOfShort);
            recordingOffset += numberOfShort;

        }


        record.stop();
        record.release();
        stopTimerThread();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mButton.setText("Recognizing...");
            }
        });

        float[] floatInputBuffer = new float[RECORDING_LENGTH];

        // feed in float values between -1.0f and 1.0f by dividing the signed 16-bit inputs.
        for (int i = 0; i < RECORDING_LENGTH; ++i) {
            floatInputBuffer[i] = recordingBuffer[i] / (float) Short.MAX_VALUE;
        }

        final String result = recognize(floatInputBuffer);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                showTranslationResult(result);
                try {
                    int length = result.length();
                    int i = 0;
                    int[] signImages = {};
                    String[] text = {};


                    try {
                        for (i = 0; i < length; i++) {
                            String val = String.valueOf(result.charAt(i));
                            signImages = Arrays.copyOf(signImages, signImages.length + 1);
                            text = Arrays.copyOf(text, text.length + 1);
                            if (val != " ") {
                                signImages[signImages.length - 1] = getResources().getIdentifier(val.toLowerCase(), "drawable", getPackageName());
                            } else {
                                val = "white";
                                signImages[signImages.length - 1] = getResources().getIdentifier(val, "drawable", getPackageName());
                            }
                            text[text.length - 1] = val;
                            System.out.println(Arrays.toString(signImages));
                            System.out.println(Arrays.toString(text));
                        }
                    } catch (Exception e) {
                        System.out.println("error message is" + e);
                    }
                    GridAdapter gridAdapter = new GridAdapter(MainActivity.this, text, signImages);
                    binding.gridView.setAdapter(gridAdapter);
                } catch (Exception e) {
                    Toast.makeText(MainActivity.this, "Please enter valid image name", Toast.LENGTH_SHORT).show();
                }
                mButton.setEnabled(true);
                mButton.setText("Start Recording");

            }
        });

    }


    private String recognize(float[] floatInputBuffer) {
        if (module == null) {
            module = LiteModuleLoader.load(assetFilePath(getApplicationContext(), "wav2vec2.ptl"));
        }

        double wav2vecinput[] = new double[RECORDING_LENGTH];
        for (int n = 0; n < RECORDING_LENGTH; n++)
            wav2vecinput[n] = floatInputBuffer[n];

        FloatBuffer inTensorBuffer = Tensor.allocateFloatBuffer(RECORDING_LENGTH);
        for (double val : wav2vecinput)
            inTensorBuffer.put((float)val);

        Tensor inTensor = Tensor.fromBlob(inTensorBuffer, new long[]{1, RECORDING_LENGTH});
        final String result = module.forward(IValue.from(inTensor)).toStr();

        return result;
    }
}