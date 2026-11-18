using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.UI;


public class TCP_Client : MonoBehaviour
{
    public static TCP_Client instance;
    TcpClient client;
    NetworkStream stream;

    private byte[] buffer = new byte[1024];

    // record the 10 (or more in furture) history of lsl (this is for other script to read)
    public Queue<int> valueHistory = new Queue<int>();
    public int historyLength = 10;
    // for simulate input
    public bool is_simulated = true;
    private float updateInterval = 0.1f; // LSL or input interval
    private float timer = 0f;
    public int simulatedInput = 0; // 0 or 1
    private GameManager GM;
    Thread receiveThread;

    // 用來存放要在主執行緒執行的任務
    private readonly Queue<Action> mainThreadActions = new Queue<Action>();

    // 還沒連上 python 時要送的訊息，連上後依序送出
    private readonly object sendLock = new object();
    private readonly List<string> pendingMessages = new List<string>();
    private const int MaxPendingMessages = 50;
    private bool pythonReady = false; // 已連線且 pending 訊息已送完 (sendLock 保護)
    private bool isConnecting = false;
    private bool destroyed = false;

    // status label (created in code)
    private Image statusIcon;
    private TextMeshProUGUI statusText;
    private int shownState = -1;

    [Header("LSL sender parameter")]
    public bool send_info_to_python = false;
    #region sigleton
    private void Awake()
    {
        historyLength = Config.PredictionHistoryLength;
        /*client = new TcpClient("127.0.0.1", 50007);
        stream = client.GetStream();*/
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of GameManager found!");
            Destroy(gameObject);
            return;
        }
        else
        {
            // DontDestroyOnLoad(gameObject);
            instance = this;
        }
    }
    #endregion
    // Start is called before the first frame update
    void Start()
    {
        if (instance != this) return;
        GM = GameManager.instance;
        CreateStatusLabel();
        StartCoroutine(ConnectLoop(Config.TCP_HOST, Config.TCP_PORT)); // 每 1 秒重試直到連上
    }

    // Update is called once per frame
    void Update()
    {
        // 每一幀在主執行緒執行佇列中的任務
        lock (mainThreadActions)
        {
            while (mainThreadActions.Count > 0)
                mainThreadActions.Dequeue().Invoke();
        }

        UpdateStatusLabel();

        if (stream == null && is_simulated)
        {
            timer += Time.deltaTime;
            if (timer >= updateInterval)
            {
                timer -= updateInterval;
                update_history(simulatedInput);
            }
        }
        else
        {
            if (send_info_to_python)
            {
                send_string_to_python("test");
                send_info_to_python = false;
            }
        }
    }
    void ReceiveLoop(NetworkStream s)
    {
        byte[] buffer = new byte[1024];
        try
        {
            SendNow(Config.send_python_tcp_model_str); // send
            FlushPendingMessages();
            while (true)
            {
                int bytes = s.Read(buffer, 0, buffer.Length);
                if (bytes <= 0) break; // python 關閉連線
                string marker = Encoding.UTF8.GetString(buffer, 0, bytes);
                if (marker == "0" || marker == "1") update_history(int.Parse(marker));  // if mark is prediction
                Debug.Log("Received from Python: " + marker);

                if (marker.Contains(Config.receive_python_tcp_model_str)) // 需要為 Config.receive_python_tcp_model_str 的內容才會觸發
                {
                    lock (mainThreadActions)
                    {
                        mainThreadActions.Enqueue(() => set_python_models_name(marker));
                    }
                }
                if (GM != null && marker.Contains(Config.calibration_done))
                {
                    GM.calibration_model_finish = true;
                }
                if (GM != null && GM.is_training && (marker != "0" && marker != "1")) // GM.is_training 在 Calibration 後面觸發或是 按下 training button 觸發
                {
                    lock (mainThreadActions) // 將需要在主執行緒執行的邏輯排入佇列，因為 thread 不能呼叫 UI 功能
                    {
                        // 這裡的程式會在主執行緒中執行
                        mainThreadActions.Enqueue(() => GM.onTrainStartCallback.Invoke(marker));
                    }
                }
            }
        }
        catch (Exception e)
        {
            if (!destroyed) Debug.LogWarning("TCP receive loop ended: " + e.Message);
        }
        HandleDisconnect(s);
    }

    // 連線中斷：清掉 stream，並在主執行緒重新連線
    void HandleDisconnect(NetworkStream s)
    {
        TcpClient oldClient;
        lock (sendLock)
        {
            if (stream != s) return; // 已經被 OnDestroy 或新的連線取代
            stream = null;
            pythonReady = false;
            oldClient = client;
        }
        try { s.Close(); } catch { }
        try { if (oldClient != null) oldClient.Close(); } catch { }
        if (destroyed) return;
        Debug.LogWarning("Connection to Python lost");
        lock (mainThreadActions)
        {
            mainThreadActions.Enqueue(() =>
            {
                is_simulated = true;
                if (GM != null) GM.use_LSL_to_controll_saber = false;
                StartCoroutine(ConnectLoop(Config.TCP_HOST, Config.TCP_PORT));
            });
        }
    }

    void set_python_models_name(string inputString)
    {
        // 找到 Config.receive_python_tcp_model_str 之後的字串
        int startIndex = inputString.IndexOf(Config.receive_python_tcp_model_str) + Config.receive_python_tcp_model_str.Length;

        // 檢查是否能找到 Config.receive_python_tcp_model_str
        if (startIndex > 0)
        {
            string modelsPart = inputString.Substring(startIndex);

            // 以 Config.separate_str 分隔字串
            string[] models = modelsPart.Split(new string[] { Config.separate_str }, StringSplitOptions.None);
            List<string> modelList = new List<string>(models);
            modelList.RemoveAt(0); // 移除第一個空值
            // 把分割後的字串加入 List
            GameDataManager.instance.python_models_name = modelList;
            if(GM.onOnPythonModelSetCallback != null)
                GM.onOnPythonModelSetCallback.Invoke();
        }
    }

    public void send_string_to_python(string message)
    {
        lock (sendLock)
        {
            if (!pythonReady) // 還沒連上 (或 pending 還沒送完)：先存起來，連上後依序送出
            {
                if (pendingMessages.Count >= MaxPendingMessages) pendingMessages.RemoveAt(0);
                pendingMessages.Add(message);
                return;
            }
        }
        SendNow(message);
    }

    void SendNow(string message)
    {
        NetworkStream s;
        lock (sendLock) { s = stream; }
        if (s == null) return;
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            lock (sendLock) { s.Write(data, 0, data.Length); }
        }
        catch (Exception e)
        {
            Debug.LogWarning("TCP send failed: " + e.Message);
        }
    }

    // 依序送出 pending 訊息 (背景執行緒)，每則間隔 50 ms 避免 python 端 recv 把多則訊息黏在一起
    void FlushPendingMessages()
    {
        while (true)
        {
            string message;
            NetworkStream s;
            lock (sendLock)
            {
                s = stream;
                if (s == null) return;
                if (pendingMessages.Count == 0)
                {
                    pythonReady = true;
                    return;
                }
                message = pendingMessages[0];
                pendingMessages.RemoveAt(0);
                byte[] data = Encoding.UTF8.GetBytes(message);
                s.Write(data, 0, data.Length);
            }
            Thread.Sleep(50);
        }
    }

    public void update_history(int value) // update history queue
    {
        // Add new value and keep the queue size
        lock (valueHistory)
        {
            valueHistory.Enqueue(value);
            if (valueHistory.Count > historyLength)
                valueHistory.Dequeue();
        }
    }

    void OnDestroy()
    {
        destroyed = true;
        NetworkStream s;
        TcpClient c;
        lock (sendLock)
        {
            s = stream;
            c = client;
            stream = null;
            pythonReady = false;
        }
        try { if (s != null) s.Close(); } catch { }
        try { if (c != null) c.Close(); } catch { }
    }

    IEnumerator ConnectLoop(string host, int port)
    {
        if (isConnecting) yield break;
        isConnecting = true;
        bool loggedFailure = false;

        while (!destroyed)
        {
            TcpClient newClient = new TcpClient();
            bool done = false;
            bool ok = false;

            // 在背景執行緒中嘗試連線
            Thread connectThread = new Thread(() =>
            {
                try
                {
                    newClient.Connect(host, port);
                    ok = newClient.Connected;
                }
                catch (Exception e)
                {
                    if (!loggedFailure) Debug.LogWarning($"❌ 連線錯誤: {e.Message}");
                }
                done = true;
            });
            connectThread.IsBackground = true;
            connectThread.Start();

            while (!done && !destroyed)
                yield return null;

            if (destroyed)
            {
                try { newClient.Close(); } catch { }
                break;
            }

            if (ok)
            {
                Debug.Log("✅ 連線成功！");
                lock (sendLock)
                {
                    client = newClient;
                    stream = newClient.GetStream();
                    pythonReady = false;
                }
                NetworkStream s = stream;
                receiveThread = new Thread(() => ReceiveLoop(s));
                receiveThread.IsBackground = true;  // 像是 python 的 daemon=True，主程序中斷，就跟者斷掉
                receiveThread.Start();
                is_simulated = false;
                if (GM != null) GM.use_LSL_to_controll_saber = true;
                break;
            }

            if (!loggedFailure)
            {
                Debug.LogWarning("⏰ 連線失敗，每 1 秒重試直到連上 Python");
                loggedFailure = true;
            }
            try { newClient.Close(); } catch { }
            yield return new WaitForSeconds(1f);
        }

        isConnecting = false;
    }

    void CreateStatusLabel()
    {
        GameObject canvasObj = new GameObject("PythonStatusCanvas", typeof(Canvas), typeof(CanvasScaler));
        canvasObj.transform.SetParent(transform, false);
        Canvas canvas = canvasObj.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 1000;
        CanvasScaler scaler = canvasObj.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);

        GameObject panel = new GameObject("Panel", typeof(RectTransform));
        panel.transform.SetParent(canvasObj.transform, false);
        RectTransform panelRect = panel.GetComponent<RectTransform>();
        panelRect.anchorMin = panelRect.anchorMax = panelRect.pivot = new Vector2(1f, 1f);
        panelRect.anchoredPosition = new Vector2(-16f, -16f);
        panelRect.sizeDelta = new Vector2(300f, 32f);

        GameObject iconObj = new GameObject("Icon", typeof(RectTransform), typeof(Image));
        iconObj.transform.SetParent(panel.transform, false);
        RectTransform iconRect = iconObj.GetComponent<RectTransform>();
        iconRect.anchorMin = iconRect.anchorMax = iconRect.pivot = new Vector2(1f, 0.5f);
        iconRect.anchoredPosition = Vector2.zero;
        iconRect.sizeDelta = new Vector2(22f, 22f);
        statusIcon = iconObj.GetComponent<Image>();
        statusIcon.sprite = CreateCircleSprite(32);
        statusIcon.raycastTarget = false;

        GameObject textObj = new GameObject("Text", typeof(RectTransform), typeof(TextMeshProUGUI));
        textObj.transform.SetParent(panel.transform, false);
        RectTransform textRect = textObj.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = new Vector2(-30f, 0f);
        statusText = textObj.GetComponent<TextMeshProUGUI>();
        if (TMP_Settings.defaultFontAsset != null) statusText.font = TMP_Settings.defaultFontAsset;
        statusText.fontSize = 22;
        statusText.alignment = TextAlignmentOptions.MidlineRight;
        statusText.raycastTarget = false;

        UpdateStatusLabel();
    }

    static Sprite CreateCircleSprite(int size)
    {
        Texture2D tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        float r = size / 2f;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float d = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), new Vector2(r, r));
                float a = Mathf.Clamp01(r - d);
                tex.SetPixel(x, y, new Color(1f, 1f, 1f, a));
            }
        }
        tex.Apply();
        return Sprite.Create(tex, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f));
    }

    void UpdateStatusLabel()
    {
        if (statusText == null || statusIcon == null) return;
        int state = stream != null ? 1 : 0;
        if (state == shownState) return;
        shownState = state;
        float alpha = (GM != null && GM.now_stage != Stage.LOBBY) ? 0.1f : 1f; // 歌曲中變淡，降低干擾
        Color c = state == 1 ? new Color(0.2f, 0.8f, 0.3f, alpha) : new Color(0.9f, 0.2f, 0.2f, alpha);
        statusIcon.color = c;
        statusText.color = c;
        statusText.text = state == 1 ? "Connected" : "Not Connected";
    }
}
