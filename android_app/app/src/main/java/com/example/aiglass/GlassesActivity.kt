package com.example.aiglass

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.xr.glimmer.Text
import androidx.xr.glimmer.surface

class GlassesActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Log.d("AIglass", "GlassesActivity onCreate")
        setContent {
            GlassHomeScreen()
        }
    }
}

@Composable
fun GlassHomeScreen() {
    Box(
        modifier = Modifier
            .surface(focusable = false)
            .fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = "안녕하세요, KSL 글래스!"
        )
    }
}
