package com.example.animesagan;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Random;

public class SaGan {
    public final static int LATENT_DIM = 100;
    private Module model;
    private static SaGan instance = null;
    private SaGan(String modelPath){
        model = Module.load(modelPath);

    }
    public static SaGan getInstance(String modelPath){
        synchronized (SaGan.class){
            if (instance == null){
                instance = new SaGan(modelPath);
            }
        }
        return instance;
    }
    public float[] getLatentVector(){
        Random rand = new Random();
        float[] vector = new float[LATENT_DIM];
        for (int i = 0; i < vector.length; i++){
            vector[i] = (float) rand.nextGaussian();
        }
        return vector;
    }
    public int[] convertToIntImage(float[] vector){
        int[] result = new int[vector.length];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = vector[i] * 0.5f + 0.5f;
            result[i] = (int)(255 * vector[i]);
        }
        return result;
    }
    public int[] getImage(){
        float[] vector = getLatentVector();

        Tensor latent = Tensor.fromBlob(vector, new long[]{1, LATENT_DIM});
        Tensor image = model.forward(IValue.from(latent)).toTensor();
        return convertToIntImage(image.getDataAsFloatArray());
    }

}
