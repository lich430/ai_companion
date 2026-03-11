package com.example.wechatglm;

import de.robv.android.xposed.IXposedHookLoadPackage;
import de.robv.android.xposed.XC_MethodHook;
import de.robv.android.xposed.XposedHelpers;
import de.robv.android.xposed.callbacks.XC_LoadPackage;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.ContentValues;
import android.os.Process;
import android.util.Log;
import java.util.HashSet;
import java.util.concurrent.TimeUnit;
import org.json.JSONObject;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * 完整版：保留消息监听 + ADB打开聊天窗口
 * 核心：主进程专属 + 只Hook一次 + 轻量数据库回调
 */
public class WechatGLMHook implements IXposedHookLoadPackage {
    private static final String TAG = "WechatGLM-8069";
    private static final String BACKEND_INCOMING_URL = "http://45.119.97.94:8080/hook/incoming";
    private static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

    // 广播相关
    private static final String ACTION_OPEN_CHAT = "com.example.wechatglm.OPEN_CHAT";
    private static final String EXTRA_WXID = "wxid";

    // 核心防护：只Hook一次 + 只在主进程执行
    private static boolean isMainProcessHooked = false;
    // 微信上下文
    private Context wechatContext = null;
    // 消息去重（轻量版）
    private final HashSet<Long> processedMsgIds = new HashSet<>();
    private final OkHttpClient okHttpClient = new OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .build();

    @Override
    public void handleLoadPackage(XC_LoadPackage.LoadPackageParam lpparam) {
        // 1. 只处理微信包名
        if (!"com.tencent.mm".equals(lpparam.packageName)) {
            return;
        }

        // 2. 只在微信主进程执行（核心：过滤所有子进程）
        String currentProcessName = getCurrentProcessName();
        if (!"com.tencent.mm".equals(currentProcessName)) {
            Log.d(TAG, "🔍 跳过微信子进程：" + currentProcessName);
            return;
        }

        // 3. 只Hook一次（核心：避免重复Hook导致数据库异常）
        if (isMainProcessHooked) {
            Log.d(TAG, "🔍 微信主进程已Hook，跳过重复执行");
            return;
        }
        isMainProcessHooked = true;

        // 打印初始化日志（只打印一次）
        Log.d(TAG, "========================");
        Log.d(TAG, "  微信主进程Hook初始化");
        Log.d(TAG, "  进程名：" + currentProcessName);
        Log.d(TAG, "========================");

        // 4. 先获取Context（主界面LauncherUI）
        hookLauncherUIForContext(lpparam);

        // 5. 再Hook数据库（只在主进程，只Hook一次）
        hookWechatDatabase(lpparam);

        Log.d(TAG, "✅ 微信主进程Hook完成（消息监听+ADB功能已加载）");
    }

    /**
     * 获取当前进程名（精准过滤微信子进程）
     */
    private String getCurrentProcessName() {
        try {
            // Android原生API获取进程名
            return (String) XposedHelpers.callStaticMethod(
                    XposedHelpers.findClass("android.app.ActivityThread", null),
                    "currentProcessName"
            );
        } catch (Exception e) {
            // 兜底：通过PID+UID标识
            return Process.myPid() + "_" + Process.myUid();
        }
    }

    /**
     * Hook微信主界面获取Context（只执行一次）
     */
    private void hookLauncherUIForContext(XC_LoadPackage.LoadPackageParam lpparam) {
        try {
            Class<?> launcherUIClass = XposedHelpers.findClass("com.tencent.mm.ui.LauncherUI", lpparam.classLoader);
            XposedHelpers.findAndHookMethod(launcherUIClass, "onCreate", android.os.Bundle.class, new XC_MethodHook() {
                @Override
                protected void afterHookedMethod(MethodHookParam param) {
                    if (wechatContext == null) {
                        wechatContext = (Context) param.thisObject;
                        Log.d(TAG, "✅ 从LauncherUI获取Context成功");
                        // 注册ADB广播接收器
                        registerChatBroadcastReceiver();
                    }
                }
            });
        } catch (Throwable e) {
            Log.e(TAG, "❌ Hook LauncherUI失败，Context兜底中...", e);
            // Context兜底：从ActivityThread获取
            try {
                Class<?> atClass = XposedHelpers.findClass("android.app.ActivityThread", lpparam.classLoader);
                Object atInstance = XposedHelpers.callStaticMethod(atClass, "currentActivityThread");
                wechatContext = (Context) XposedHelpers.getObjectField(atInstance, "mInitialApplication");
                Log.d(TAG, "✅ 兜底获取Context成功");
                registerChatBroadcastReceiver();
            } catch (Exception e2) {
                Log.e(TAG, "❌ Context获取失败，ADB功能不可用（消息监听仍可用）", e2);
            }
        }
    }

    /**
     * Hook微信数据库（只监听message表，轻量无阻塞）
     */
    private void hookWechatDatabase(XC_LoadPackage.LoadPackageParam lpparam) {
        try {
            Class<?> sqliteDbClass = XposedHelpers.findClass("com.tencent.wcdb.database.SQLiteDatabase", lpparam.classLoader);

            // Hook insert方法（监听消息入库）
            XposedHelpers.findAndHookMethod(
                    sqliteDbClass,
                    "insert",
                    String.class,
                    String.class,
                    ContentValues.class,
                    new MessageHookCallback()
            );

            // Hook insertWithOnConflict方法（兜底）
            XposedHelpers.findAndHookMethod(
                    sqliteDbClass,
                    "insertWithOnConflict",
                    String.class,
                    String.class,
                    ContentValues.class,
                    int.class,
                    new MessageHookCallback()
            );

            Log.d(TAG, "✅ 微信数据库Hook成功（监听message表）");
        } catch (Throwable e) {
            Log.e(TAG, "❌ Hook微信数据库失败", e);
        }
    }

    /**
     * 注册ADB广播接收器（只在主进程注册一次）
     */
    private void registerChatBroadcastReceiver() {
        if (wechatContext == null) {
            Log.e(TAG, "❌ Context为空，无法注册广播");
            return;
        }

        try {
            BroadcastReceiver chatReceiver = new BroadcastReceiver() {
                @Override
                public void onReceive(Context context, Intent intent) {
                    String wxid = intent.getStringExtra(EXTRA_WXID);
                    if (wxid == null || wxid.isEmpty()) {
                        Log.e(TAG, "❌ ADB广播未传递wxid参数");
                        return;
                    }
                    Log.d(TAG, "✅ 接收到ADB广播，打开wxid：" + wxid);
                    openChatWindow(wxid);
                }
            };

            IntentFilter filter = new IntentFilter(ACTION_OPEN_CHAT);
            // 兼容Android 12+和低版本
            try {
                wechatContext.registerReceiver(chatReceiver, filter, Context.RECEIVER_EXPORTED);
            } catch (Exception e) {
                wechatContext.registerReceiver(chatReceiver, filter);
            }
            Log.d(TAG, "✅ ADB广播接收器注册成功");
        } catch (Exception e) {
            Log.e(TAG, "❌ 注册ADB广播失败", e);
        }
    }

    /**
     * 打开聊天窗口（稳定Intent方案）
     */
    private void openChatWindow(String wxid) {
        try {
            Intent chatIntent = new Intent();
            chatIntent.setClassName("com.tencent.mm", "com.tencent.mm.ui.chatting.ChattingUI");
            chatIntent.putExtra("Chat_User", wxid);
            chatIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            wechatContext.startActivity(chatIntent);
            Log.d(TAG, "✅ 成功打开wxid=" + wxid + "的聊天窗口");
        } catch (Exception e) {
            Log.e(TAG, "❌ 打开聊天窗口失败", e);
        }
    }

    /**
     * 消息监听回调（轻量版：只做必要逻辑，不阻塞数据库）
     */
    private class MessageHookCallback extends XC_MethodHook {
        @Override
        protected void afterHookedMethod(MethodHookParam param) {
            try {
                // 1. 只处理message表（过滤其他表，减少性能消耗）
                String tableName = (String) param.args[0];
                if (tableName == null || !"message".equals(tableName)) {
                    return;
                }

                // 2. 获取消息内容（只解析必要字段）
                ContentValues cv = (ContentValues) param.args[2];
                if (cv == null) return;

                long msgId = cv.getAsLong("msgId");
                String content = cv.getAsString("content");
                String talker = cv.getAsString("talker");
                int isSend = cv.getAsInteger("isSend");
                int type = cv.getAsInteger("type");

                // 3. 只处理接收的文本消息（过滤发送的/非文本消息）
                if (isSend != 0 || type != 1 || content == null || content.isEmpty()) {
                    return;
                }

                // 4. 消息去重（轻量HashSet，避免重复处理）
                if (processedMsgIds.contains(msgId)) {
                    return;
                }
                processedMsgIds.add(msgId);
                // 限制HashSet大小，避免内存泄漏
                if (processedMsgIds.size() > 2000) {
                    processedMsgIds.clear();
                }

                // 5. 打印消息日志（核心需求）
                Log.d(TAG, "✅ 捕获新消息 | 发送者：" + talker + " | 内容：" + content);

                // 6. 异步处理消息（不阻塞数据库读写）
                new Thread(() -> {
                    try {
                        pushIncomingToBackend(talker, content);
                    } catch (Exception e) {
                        Log.e(TAG, "❌ 消息异步处理失败", e);
                    }
                }).start();

            } catch (Throwable e) {
                // 捕获所有异常，避免影响微信数据库正常读写
                Log.e(TAG, "❌ 消息监听回调异常（不影响微信运行）", e);
            }
        }
    }

    /**
     * 推送新消息到后端：后端会入记忆、请求GLM并把reply入内存队列
     */
    private void pushIncomingToBackend(String username, String message) {
        String user = username == null ? "" : username.trim();
        String msg = message == null ? "" : message.trim();
        if (user.isEmpty() || msg.isEmpty()) {
            return;
        }

        try {
            JSONObject body = new JSONObject();
            body.put("username", user);
            body.put("user_id", user);
            body.put("message", msg);

            Request request = new Request.Builder()
                    .url(BACKEND_INCOMING_URL)
                    .post(RequestBody.create(body.toString(), JSON))
                    .build();

            try (Response response = okHttpClient.newCall(request).execute()) {
                String respBody = response.body() != null ? response.body().string() : "";
                Log.d(TAG, "✅ pushIncomingToBackend code=" + response.code() + " body=" + respBody);
            }
        } catch (Exception e) {
            Log.e(TAG, "❌ pushIncomingToBackend失败", e);
        }
    }
}
