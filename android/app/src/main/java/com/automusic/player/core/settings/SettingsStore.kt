package com.automusic.player.core.settings

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import org.json.JSONArray
import org.json.JSONObject

private val Context.dataStore by preferencesDataStore(name = "settings")

/** 模型供应商(OpenAI 兼容多模态接口)。 */
data class Provider(
    val name: String,
    val baseUrl: String,
    val apiKey: String,
    val model: String,
)

private val DEFAULT_PROVIDER = Provider(
    name = "774966 测试渠道",
    baseUrl = "https://api.774966.xyz/v1",
    apiKey = "sk-zAbqTwB7SUsdQ6mFT0lImnWf98Z5QU6255Qr73WxJpBxcFCG",
    model = "gpt-6-astra",
)

/** 设置存储:供应商列表 + 激活项,JSON 持久化(协议与桌面版一致)。 */
class SettingsStore(private val context: Context) {

    private val KEY_PROVIDERS = stringPreferencesKey("providers")
    private val KEY_ACTIVE = stringPreferencesKey("active_provider")

    val providersFlow: Flow<List<Provider>> = context.dataStore.data.map { prefs ->
        providers(prefs[KEY_PROVIDERS])
    }

    val activeNameFlow: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[KEY_ACTIVE] ?: if (prefs[KEY_PROVIDERS] == null) DEFAULT_PROVIDER.name else ""
    }

    suspend fun getActiveProvider(): Provider? {
        val prefs = context.dataStore.data.first()
        val active = prefs[KEY_ACTIVE] ?: if (prefs[KEY_PROVIDERS] == null) DEFAULT_PROVIDER.name else return null
        return providers(prefs[KEY_PROVIDERS]).firstOrNull { it.name == active }
    }

    suspend fun save(provider: Provider, previousName: String? = null, makeActive: Boolean = false) {
        context.dataStore.edit { prefs ->
            val currentActive = prefs[KEY_ACTIVE]
                ?: if (prefs[KEY_PROVIDERS] == null) DEFAULT_PROVIDER.name else ""
            val list = providers(prefs[KEY_PROVIDERS]).toMutableList()
            val oldName = previousName ?: provider.name
            val idx = list.indexOfFirst { it.name == oldName }
            if (idx >= 0) list[idx] = provider else list.add(provider)
            prefs[KEY_PROVIDERS] = encode(list)
            if (makeActive || currentActive == oldName) prefs[KEY_ACTIVE] = provider.name
        }
    }

    suspend fun delete(name: String) {
        context.dataStore.edit { prefs ->
            val list = providers(prefs[KEY_PROVIDERS]).filter { it.name != name }
            prefs[KEY_PROVIDERS] = encode(list)
            if (prefs[KEY_ACTIVE] == name) prefs[KEY_ACTIVE] = ""
        }
    }

    suspend fun setActive(name: String) {
        context.dataStore.edit { prefs -> prefs[KEY_ACTIVE] = name }
    }

    private fun encode(list: List<Provider>): String {
        val arr = JSONArray()
        for (p in list) {
            arr.put(
                JSONObject()
                    .put("name", p.name)
                    .put("baseUrl", p.baseUrl)
                    .put("apiKey", p.apiKey)
                    .put("model", p.model)
            )
        }
        return arr.toString()
    }

    private fun providers(stored: String?): List<Provider> =
        if (stored == null) {
            listOf(DEFAULT_PROVIDER)
        } else {
            decode(stored).map { provider ->
                if (
                    provider.name == DEFAULT_PROVIDER.name &&
                    provider.baseUrl.trimEnd('/') == DEFAULT_PROVIDER.baseUrl &&
                    provider.model == LEGACY_DEFAULT_MODEL
                ) {
                    provider.copy(model = DEFAULT_PROVIDER.model)
                } else {
                    provider
                }
            }
        }

    private fun decode(json: String): List<Provider> = try {
        val arr = JSONArray(json)
        buildList {
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                add(
                    Provider(
                        name = o.optString("name"),
                        baseUrl = o.optString("baseUrl"),
                        apiKey = o.optString("apiKey"),
                        model = o.optString("model"),
                    )
                )
            }
        }
    } catch (e: Exception) {
        emptyList()
    }

    private companion object {
        const val LEGACY_DEFAULT_MODEL = "gpt-5.6-sol"
    }
}
