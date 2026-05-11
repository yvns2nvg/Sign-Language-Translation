package com.example.aiglass

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.xr.projected.ProjectedContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Button(onClick = {
                    Log.d("AIglass", "Starting GlassesActivity...")
                    try {
                        val options = ProjectedContext.createProjectedActivityOptions(this@MainActivity)
                        val intent = Intent(this@MainActivity, GlassesActivity::class.java)
                        startActivity(intent, options.toBundle())
                        Log.d("AIglass", "startActivity called successfully")
                    } catch (e: Exception) {
                        Log.e("AIglass", "Failed to start GlassesActivity", e)
                    }
                }) {
                    Text("AI 글래스 화면 켜기")
                }
            }
        }
    }
}
