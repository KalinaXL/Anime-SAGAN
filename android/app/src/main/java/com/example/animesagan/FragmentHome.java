package com.example.animesagan;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

public class FragmentHome extends Fragment {
    private Button btnGetImage;
    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        super.onCreateView(inflater, container, savedInstanceState);
        return inflater.inflate(R.layout.fragment_home, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        btnGetImage = view.findViewById(R.id.btn_get_image);
        btnGetImage.setOnClickListener(v -> {
            getParentFragmentManager().beginTransaction()
                                    .addToBackStack(null)
                                    .replace(R.id.nav_host_fragment, new FragmentImage(), "Tag")
                                    .commit();
        });
    }
}
