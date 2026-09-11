package com.automusic.player.core.recognizer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Base64
import com.automusic.player.core.Prompt
import com.automusic.player.core.settings.Provider
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit

/**
 * OpenAI 兼容协议的多模态识别接口(通义/GLM/Kimi/OpenAI 通用)。
 * 图片发送前自动等比压缩(长边 1600px、JPEG q85);流式响应防长推理读超时。
 */
class ScoreRecognizer(
    private val apiBase: String,
    private val apiKey: String,
    private val model: String,
    private val readTimeoutS: Long = 300L,
) {

    suspend fun recognizeImage(context: Context, uri: Uri): String =
        withContext(Dispatchers.IO) {
            val (bytes, mime) = compressImage(context, uri)
            chat(Prompt.JIANPU_PROMPT, bytes, mime)
        }

    suspend fun recognizeDocument(text: String): String = withContext(Dispatchers.IO) {
        chat("${Prompt.JIANPU_PROMPT}\n\n以下是文档提取出的乐谱内容:\n$text", null, null)
    }

    /** 仅查询模型列表验证地址与密钥，不消耗生成额度。 */
    suspend fun testConnection(): String = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url("${apiBase.trimEnd('/')}/models")
            .header("Authorization", "Bearer $apiKey")
            .get()
            .build()
        httpClient().newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                val detail = response.body?.string().orEmpty().take(200)
                throw RuntimeException("接口返回错误(HTTP ${response.code}): $detail")
            }
            val body = response.body?.string().orEmpty()
            runCatching { JSONObject(body) }
                .getOrElse { throw RuntimeException("模型列表响应不是 JSON，请检查 Base URL 是否包含 /v1") }
            "HTTP ${response.code}"
        }
    }

    private fun chat(textPrompt: String, image: ByteArray?, mime: String?): String {
        val content = JSONArray()
            .put(JSONObject().put("type", "text").put("text", textPrompt))
        if (image != null && mime != null) {
            val b64 = Base64.encodeToString(image, Base64.NO_WRAP)
            content.put(
                JSONObject()
                    .put("type", "image_url")
                    .put("image_url", JSONObject().put("url", "data:$mime;base64,$b64"))
            )
        }

        val payload = JSONObject()
            .put("model", model)
            .put("messages", JSONArray().put(JSONObject().put("role", "user").put("content", content)))
            .put("stream", true)

        val request = Request.Builder()
            .url("${apiBase.trimEnd('/')}/chat/completions")
            .header("Authorization", "Bearer $apiKey")
            .post(payload.toString().toRequestBody("application/json".toMediaType()))
            .build()

        val client = httpClient()

        client.newCall(request).execute().use { resp ->
            if (!resp.isSuccessful) {
                val detail = resp.body?.string().orEmpty().take(300)
                throw RuntimeException("接口返回错误(HTTP ${resp.code}): $detail")
            }
            val source = resp.body?.source() ?: throw RuntimeException("响应为空")
            val parts = StringBuilder()
            while (true) {
                val line = source.readUtf8Line() ?: break
                if (!line.startsWith("data:")) continue
                val data = line.removePrefix("data:").trim()
                if (data == "[DONE]") break
                try {
                    val chunk = JSONObject(data)
                    val choices = chunk.optJSONArray("choices") ?: continue
                    if (choices.length() == 0) continue
                    val delta = choices.getJSONObject(0).optJSONObject("delta") ?: continue
                    parts.append(delta.optString("content"))
                } catch (e: Exception) {
                    continue
                }
            }
            val text = parts.toString().trim()
            if (text.isEmpty()) {
                throw RuntimeException("模型未返回内容:请检查模型名称是否为支持图片输入的视觉模型")
            }
            return text
        }
    }

    private fun httpClient(): OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(readTimeoutS, TimeUnit.SECONDS)
        .build()

    private fun compressImage(context: Context, uri: Uri): Pair<ByteArray, String> {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw RuntimeException("无法读取图片")
        }

        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= IMAGE_MAX_SIDE) {
            sample *= 2
        }
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val bitmap = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        } ?: throw RuntimeException("无法读取图片")

        var bmp = bitmap
        val curMax = maxOf(bmp.width, bmp.height)
        if (curMax > IMAGE_MAX_SIDE) {
            val scale = IMAGE_MAX_SIDE.toFloat() / curMax
            bmp = Bitmap.createScaledBitmap(
                bmp,
                (bmp.width * scale).toInt().coerceAtLeast(1),
                (bmp.height * scale).toInt().coerceAtLeast(1),
                true,
            )
        }

        val buf = ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, buf)
        if (bmp !== bitmap) bitmap.recycle()
        return buf.toByteArray() to "image/jpeg"
    }

    companion object {
        private const val IMAGE_MAX_SIDE = 1600
        private const val JPEG_QUALITY = 85
    }
}

object RecognizerFactory {
    /** 按激活供应商创建识别器;无有效配置时返回 null，由界面明确引导。 */
    fun create(provider: Provider?): ScoreRecognizer? =
        provider
            ?.takeIf { it.baseUrl.isNotBlank() && it.apiKey.isNotBlank() && it.model.isNotBlank() }
            ?.let { ScoreRecognizer(it.baseUrl, it.apiKey, it.model) }
}
