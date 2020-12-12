package com.example.animesagan;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import java.io.IOException;

public class FragmentImage extends Fragment {
    private SaGan saGan;
    private int[] image;
    private Bitmap bitmap, final_bm;
    private ImageView ivAnime;
    public FragmentImage(){
        bitmap = Bitmap.createBitmap(64, 64, Bitmap.Config.ARGB_8888);
    }
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        super.onCreateView(inflater, container, savedInstanceState);
        try {
            saGan = SaGan.getInstance(Utils.assertFilePath(requireContext(), "anime_sagan.pt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        image = saGan.getImage();
//        bitmap.setPixels(image, 0, 64, 0, 64, 0, 64);
        convertIntArrayToBitmap(image);
        return inflater.inflate(R.layout.fragment_image, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        ivAnime = view.findViewById(R.id.iv_anime);
        ivAnime.setImageBitmap(final_bm);
    }

    void convertIntArrayToBitmap(int[] image){
        int padding = 64 * 64;
        int r, g, b, idx;
        int c;
//        for (int i = 0; i < 64; i++){
//            for (int j = 0; j < 64; j++){
//                idx = i * 64 + j;
//                r = image[idx];
//                g = image[padding + idx];
//                b = image[2 * padding + idx];
//                r = (r << 16) & 0x00FF0000;
//                g = (g << 8) & 0x0000FF00;
//                b = b & 0x000000FF;
//
//                c = 0xFF000000 | b | g | b;
//                bitmap.setPixel(j, i, c);
//            }
//        }
        for (int i = 0; i < 64; i++){
            for (int j = 0; j < 64; j++){
                idx = i * 64 + j;
                r = image[idx];
                g = image[padding + idx];
                b = image[2 * padding + idx];
                r = (r << 16) & 0x00FF0000;
                g = (g << 8) & 0x0000FF00;
                b = b & 0x000000FF;

                c = 0xFF000000 | b | g | b;
                bitmap.setPixel(j, i, c);
            }
        }
        final_bm = Bitmap.createScaledBitmap(bitmap, 512, 512, true);
    }
}
