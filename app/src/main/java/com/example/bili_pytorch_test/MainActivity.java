package com.example.bili_pytorch_test;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class MainActivity extends AppCompatActivity {
    /**
     * 声明一些变量和常量
     */
    //用于测试的TAG
    public String TAG = "MainActivityTest";
    // 请求权限的请求码
    private static final int REQUIRE_PERMISSION_CODE = 10086;
    //开启相册的请求码
    private static final int START_ALBUM_CODE = 10080;
    //开启相机的请求码
    private static final int START_CAMERA_CODE = 10081;
    //开启摄像头按钮对象
    private Button BtnCamera;
    //开启相册按钮对象
    private Button BtnAlbum;
    private Button BtnLoad;
    //UI中的图片对象
    private ImageView imageView;
    private ImageView imageView2;
//    private ImageView imageView3;
    //UI中的文字对象
    private TextView textView;
    private TextView textView2;
    private TextView textView3;
    //模型
    Module module = null;
    //处理的Bitmap对象
    Bitmap processedBitmap = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestPermissions();
        setContentView(R.layout.ui_test1);
        //将已经声明的对象和Xml文件中的对象绑定
        imageView = findViewById(R.id.image);
//        imageView3 = findViewById(R.id.image_title);
        textView = findViewById(R.id.text);
        textView2 = findViewById(R.id.textView3);
        textView3 = findViewById(R.id.textView7);
        BtnCamera = findViewById(R.id.button_camera);
        BtnAlbum = findViewById(R.id.button_album);
        BtnLoad = findViewById(R.id.button_load);
//        imageView3.setImageDrawable(getDrawable(R.drawable.title_b));
        BtnLoad.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(MainActivity.this, "测试照片导入成功", Toast.LENGTH_SHORT).show();
                loadPics();
            }
        });
        //对开启相册按钮设定点击事件
        BtnAlbum.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent pickPhotoIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(pickPhotoIntent, START_ALBUM_CODE);
            }
        });
        //对开启相机按钮设定点击事件
        BtnCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(takePictureIntent, START_CAMERA_CODE);
            }
        });
    }

    /**
     * 此方法用于判断图片是否需要反色操作
     * 其中调用MEAN.pt模型
     * 将图片的张量取均值
     * 如果均值大于128
     * 将图片反色
     *
     * @param inputBitmap 输入的图片
     * @return 返回布尔值决定是否反色
     */
    public Boolean judgeInverse(Bitmap inputBitmap) {
        //调用模型
        try {
            module = Module.load(assetFilePath(this, "MEAN.pt"));
        } catch (IOException e) {
            Toast.makeText(this, "判断模型加载失败", Toast.LENGTH_SHORT).show();
            throw new RuntimeException(e);
        }

        float[] meanRGB = {0.0f, 0.0f, 0.0f};
        float[] stdRGB = {1.0f, 1.0f, 1.0f};

        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(inputBitmap, meanRGB, stdRGB);
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        //判断均值是否大于128
        Float mean = scores[0] * 255;
        Toast.makeText(this, "均值为" + mean, Toast.LENGTH_SHORT).show();
        if (mean >= 64) {
//            Toast.makeText(this, "判断为阳", Toast.LENGTH_SHORT).show();
            return true;
        }
        return false;
    }

    /**
     * 此方法用于处理图片大小
     * 由于用于训练的训练集都是28*28的图片，需要将输入的图片分辨率转成28*28
     *
     * @param inputBitmap 输入的图片
     * @return 返回处理后的图片
     */
    public Bitmap setSize(Bitmap inputBitmap) {
        // 创建一个28x28的Bitmap
        Bitmap outputBitmap = Bitmap.createScaledBitmap(inputBitmap, 28, 28, true);
        return outputBitmap;
    }

    /**
     * 此方法用于将图片反色
     * 实际操作是读取图片中的每个像素值，计算与256的插值
     *
     * @param inputBitmap 输入的图片
     * @return 返回处理后的图片
     */
    public Bitmap inverseColor(Bitmap inputBitmap) {
        Bitmap outputBitmap = inputBitmap.copy(inputBitmap.getConfig(), true);
        int[] pixels = new int[28 * 28];
        outputBitmap.getPixels(pixels, 0, 28, 0, 0, 28, 28);
        for (int i = 0; i < pixels.length; i++) {
            int r = Color.red(pixels[i]);
            int g = Color.green(pixels[i]);
            int b = Color.blue(pixels[i]);
            int a = Color.alpha(pixels[i]);
            pixels[i] = Color.argb(a, 255 - r, 255 - g, 255 - b);
        }
        outputBitmap.setPixels(pixels, 0, 28, 0, 0, 28, 28);
        return outputBitmap;
    }

    /**
     * 此方法是调用训练完成的手写数字神经网络模型识别输入的图片
     * 并将识别完的值显示在屏幕上
     *
     * @param bitmap 输入用于识别的图片
     */
    public void predict(Bitmap bitmap) {
        Log.d(TAG, "predict: predict");
        try {
            module = Module.load(assetFilePath(this, "MNIST.pt"));
        } catch (IOException e) {
            Toast.makeText(this, "识别模型加载失败", Toast.LENGTH_SHORT).show();
            throw new RuntimeException(e);
        }
        Tensor inputTensor = UtilsFunctions.bitmapToFloat32Tensor(bitmap);
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        Log.d(TAG, "onCreate: " + Arrays.toString(scores));
        String result = "";
//        result += "图片识别为" + Float.toString(numProcessor(scores));
        result += Float.toString(numProcessor(scores));
        textView.setText(result);
    }

    /**
     * 此方法用于将网络输出的数值进行处理
     * 识别网络的输出值是dim=1，length=10的数组
     * 将数组中的数值进行大小比较
     * 返回最大值的索引
     * 即是网络所识别出的数字
     *
     * @param output 识别网络输出的数组
     * @return 返回数组的最大值的索引
     */
    public int numProcessor(float[] output) {
        float max = output[0];
        int maxidx = 0;
        for (int i = 0; i < 10; i++) {
            if (output[i] >= max) {
                max = (int) output[i];
                maxidx = i;
            }
        }
        Log.d(TAG, "numProcessor: " + maxidx);
        return maxidx;
    }

    public Bitmap contrastAndBrightnessProcessor(Bitmap bitmap, float contrast, float brightness) {
        // 创建一个ColorMatrix对象，并设置对比度和亮度的矩阵
        ColorMatrix cm = new ColorMatrix(new float[]{
                contrast, 0, 0, 0, brightness,
                0, contrast, 0, 0, brightness,
                0, 0, contrast, 0, brightness,
                0, 0, 0, 1, 0
        });

        // 创建一个ColorMatrixColorFilter对象，并将其应用到Paint对象中
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(cm);
        Paint paint = new Paint();
        paint.setColorFilter(filter);

        // 创建一个新的Bitmap对象，并将原始Bitmap对象绘制到其中
        Bitmap result = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(result);
        canvas.drawBitmap(bitmap, 0, 0, paint);

        return result;
    }


    /**
     * 读取在app目录中的assets文件夹中的文件的绝对路径，用于读取模型文件
     *
     * @return 返回绝对的文件路径
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
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
        }
    }

    /**
     * 此方法用于获取软件需要的权限
     * 由于使用了手机的相机和相册，需要CAMERA，READ_EXTERNAL_STORAGE，WRITE_EXTERNAL_STORAGE三个权限
     * 即相机的读写储存的权限
     */
    private void requestPermissions() {
        // 定义容器，存储我们需要申请的权限
        List<String> permissionList = new ArrayList<>();

        // 检测应用是否具有CAMERA的权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.CAMERA);
        }
        // 检测应用是否具有READ_EXTERNAL_STORAGE权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        }

        // 检测应用是否具有WRITE_EXTERNAL_STORAGE权限
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }

        // 如果permissionList不为空，则说明前面检测的三种权限中至少有一个是应用不具备的
        // 则需要向用户申请使用permissionList中的权限
        if (!permissionList.isEmpty()) {
            ActivityCompat.requestPermissions(this, permissionList.toArray(new String[permissionList.size()]), REQUIRE_PERMISSION_CODE);
        }
    }

    /**
     * @param requestCode  定义的权限请求码，用于判断所求情的权限是哪一个
     * @param permissions  所请求的权限
     * @param grantResults 未拥有的权限数量
     */

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // 判断请求码
        switch (requestCode) {
            // 如果请求码是我们设定的权限请求代码值，则执行下面代码
            case REQUIRE_PERMISSION_CODE:
                if (grantResults.length > 0) {
                    for (int i = 0; i < grantResults.length; i++) {
                        // 如果请求被拒绝，则弹出下面的Toast
                        if (grantResults[i] == PackageManager.PERMISSION_DENIED) {
                            Toast.makeText(this, "申请" + permissions[i] + "被拒绝", Toast.LENGTH_SHORT).show();
                        }
                    }
                }
                break;
        }
    }

    /**
     * 此方法用于返回相机和相册所返回的图片
     * 并将其转换为Btmap格式，调用
     *
     * @param requestCode The integer request code originally supplied to
     *                    startActivityForResult(), allowing you to identify who this
     *                    result came from.
     * @param resultCode  The integer result code returned by the child activity
     *                    through its setResult().
     * @param data        An Intent, which can return result data to the caller
     *                    (various data can be attached to Intent "extras").
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {


        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == START_CAMERA_CODE && data != null) {
                // 处理相机返回的图片
                Bundle extras = data.getExtras();
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                finalNum(imageBitmap);
            } else if (requestCode == START_ALBUM_CODE && data != null) {
                // 处理相册返回的图片
                Uri selectedImageUri = data.getData();
                try {
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
                    finalNum(bitmap);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }

    /**
     * 最终对图片处理，并显示图片
     *
     * @param inputBitmap 输入的图片
     */
    public void finalNum(Bitmap inputBitmap) {
        imageView.setImageBitmap(inputBitmap);

        textView3.setVisibility(View.GONE);
        textView2.setVisibility(View.VISIBLE);
        if (judgeInverse(inputBitmap)) {
            processedBitmap = setSize(inputBitmap);
            inputBitmap = contrastAndBrightnessProcessor(inputBitmap, 1.4F, 0);
            processedBitmap = inverseColor(processedBitmap);
        } else {
            processedBitmap = setSize(inputBitmap);
        }
//        imageView.setImageBitmap(processedBitmap);
        predict(processedBitmap);
    }

    public void loadPics() {
        for (int i = 0; i < 10; i++) {
            BitmapDrawable drawable = (BitmapDrawable) getResources().getDrawable(getResources().getIdentifier("test" + i, "drawable", getPackageName()));
            Bitmap bitmap = drawable.getBitmap();
            String path = MediaStore.Images.Media.insertImage(getContentResolver(), bitmap, "ImageForTest" + i, null);
        }
    }
}