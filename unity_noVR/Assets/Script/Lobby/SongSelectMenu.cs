using System.Collections;
using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Newtonsoft.Json;
using System.Threading;
[System.Serializable]
public class LevelData
{
    public string level;
}
public class SongSelectMenu : MonoBehaviour
{
    public static SongSelectMenu instance;
    [Header("selected song")]
    public Song song;
    [Header("Main Menu Buttons")]
    public Button startButton;
    [Header("use in Song UI shower")]
    public Song_UI_shower last_selected_song;
    public GameObject level_parent_obj; // �� Song UI shower �i�H�]�w���� (�� child set Active) LevelM
    public Image targetImage;
    public TextMeshProUGUI title;
    public TextMeshProUGUI sub_title;
    public TextMeshProUGUI song_time;
    public TextMeshProUGUI cube_num;
    public TextMeshProUGUI condition_text;
    [Header("use in Song UI shower")]
    public GameObject fill_obj;  // ���F���� fill ����m�A���ϥΪ̥i�H�ݨ�ثe��ܪ�����
    private Vector3 fill_obj_pos;  // ���F���� fill ����m�A���ϥΪ̥i�H�ݨ�ثe��ܪ�����
    public AudioSource audioSource;  // �s�{�b���񪺤��e // �b Song_UI_shower.cs ����
    public AudioSource nextAudioSource;  // �U�@�Ӽ��񪺤��e // �b Song_UI_shower.cs ���� // �ثe�S���Ψ�
    public LevelData levelData = new LevelData { level = "MI" };  // �����C���Ҧ��A�b Song_UI_shower select ���ɭԷ|�]�w�� MI�A��Ҧ��b�U�誺 button Ĳ�o�|�令 button �W��
    private TCP_Client sender;

    void Awake()
    {
        if (instance == null)
            instance = this;
        else
        {
            Destroy(gameObject);
            return;
        }
        audioSource.volume = Config.volume;  
        // DontDestroyOnLoad(gameObject);
        fill_obj_pos = fill_obj.GetComponent<RectTransform>().localPosition;
    }
    
    // Start is called before the first frame update
    void Start()
    {
        //Hook events
        startButton.onClick.AddListener(StartGame);
        save_game_level();
        sender = TCP_Client.instance;
        get_condition(condition_text);
    }

    // �����s�Q�I����Ĳ�o�A�ñN Button �ǻ�����k // �D�n�� ������� level �� button �ϥ�
    public void OnLevelButtonClick(Button clickedButton)
    {
        // ������s����m
        Vector3 buttonPosition = clickedButton.transform.localPosition;
        levelData.level = clickedButton.gameObject.name;

        int index = GetObjectsIdbyName(levelData.level, level_parent_obj);
        get_note_num(index);

        // �N UI �������ʨ���s����m // ��ʽվ�
        RectTransform uiRectTransform = fill_obj.GetComponent<RectTransform>();
        Vector3 offset = new Vector3(84, -51, 0);
        uiRectTransform.localPosition = buttonPosition + offset;
        save_game_level();
    }

    public void save_game_level()
    {
        string json = JsonUtility.ToJson(levelData);
        string fullPath = Path.Combine(Application.persistentDataPath, Config.level_json_file);
        File.WriteAllText(fullPath, json);
    }
    public void reset_fill_position()
    {
        fill_obj.GetComponent<RectTransform>().localPosition = fill_obj_pos;
    }

    public void change_condition(TextMeshProUGUI textMeshProUGUI)
    {
        Config.adaptive_model = !Config.adaptive_model;
        get_condition(textMeshProUGUI);
    }

    public void get_condition(TextMeshProUGUI textMeshProUGUI)
    {
        if (Config.adaptive_model)
            textMeshProUGUI.text = "Adaptive";
        else
            textMeshProUGUI.text = "Static";
    }

    public void StartGame()
    {
        if(levelData.level == Config.level_ui_strings[(int)Level.MI])
        {
            sender.send_string_to_python(song.name);

            if (song.name == Config.Calibration) GameDataManager.instance.is_calibration = true;    

            SceneLoaderManager.instance.LoadMIStage();
            /*if (song.name == Config.Calibration) SceneLoaderManager.instance.LoadCalibrationStage();
            else SceneLoaderManager.instance.LoadMIStage();*/
        }
        else
        {
            // for VR mode
            // SceneLoaderManager.instance.LoadBeatSaberStage();
        }
    }
    int GetObjectsIdbyName(string names, GameObject p_obj)
    {
        int count = 0;
        // �T�O PObject �U���b names �}�C��������Q����
        foreach (Transform child in p_obj.transform)
        {
            if (child.gameObject.activeSelf)
            {
                if (names == child.name) break;
                count += 1;
            }
        }
        return count;
    }
    public void get_note_num(int index) // index �N�����������ӡA������סAŪ���ݹ����� note �ƶq
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, song.dir, song.file_names[index]);
        StartCoroutine(StreamingAssetLoader.ReadFileFromJar(filePath, (json) => {
            if (json != null)
            {
                OldBeatmap beatmap = JsonConvert.DeserializeObject<OldBeatmap>(json);
                if (beatmap == null || beatmap.notes == null)
                {
                    Debug.LogError("Failed to deserialize .dat file or _notes is null");
                    if (index == 0) cube_num.text = "MI trial: 0";
                    else cube_num.text = "Cubes: 0";
                }
                // if (index == 0) cube_num.text = "MI trial: " + beatmap.notes.Count.ToString();
                // if (song.name == Config.Calibration) cube_num.text = "MI trial: 20";
                // else if (index == 0) cube_num.text = "MI trial: " + ((int)(beatmap.notes.Count / (Config.group_note_num + 1)) + 1).ToString();
                if (song.name == Config.Calibration) cube_num.text = "MI trial: 48";
                // else if (index == 0) cube_num.text = "MI trial: " + ((int)(beatmap.notes.Count / (Config.group_note_num))).ToString();  // 40
                else if (index == 0) cube_num.text = "MI trial: 48";// + ((int)(beatmap.notes.Count / (Config.group_note_num))).ToString();  // 40
                else cube_num.text = "Cubes: " + beatmap.notes.Count.ToString();

                /*BeatmapData data = JsonConvert.DeserializeObject<BeatmapData>(json);
                if (data == null || data.colorNotes == null)  // use old beat map format
                {
                    OldBeatmap beatmap = JsonConvert.DeserializeObject<OldBeatmap>(json);
                    if (beatmap == null || beatmap.notes == null)
                    {
                        Debug.LogError("Failed to deserialize .dat file or _notes is null");
                        if (index == 0) cube_num.text = "MI trial: 0";
                        else cube_num.text = "Cubes: 0";
                    }
                    if (index == 0) cube_num.text = "MI trial: " + beatmap.notes.Count.ToString();
                    else cube_num.text = "Cubes: " + beatmap.notes.Count.ToString();
                }
                else
                {
                    if (index == 0) cube_num.text = "MI trial: " + data.colorNotes.Count.ToString();
                    else cube_num.text = "Cubes: " + data.colorNotes.Count.ToString();
                }*/
            }
        }));
    }
    // �� BeatmapSpawner.cs �̭��� GroupNotesByRules�Afunction �����A�令�u�^�� int �� function�A�é�e������ܡA�ثe�S�Ψ�
    public int GetGroupCountByRules(List<OldNote> sourceNotes, int n, float bpm)
    {
        float beatsPerSecond = bpm / 60f;

        float beatThreshold_up = (0.6f * Config.group_note_num + 2) * beatsPerSecond;
        float beatThreshold_down = (0.6f * Config.group_note_num) * beatsPerSecond;
        float beatThreshold_1s = 1.0f * beatsPerSecond;
        // �p�G�S����ơA�^�� 0
        if (sourceNotes == null || sourceNotes.Count == 0) return 0;
        if (n <= 0) n = 1;

        // --- ���� 1: �p�G n = 1�A�C�� Note ���O�@�աA�����^���`�� ---
        if (n == 1)
        {
            return sourceNotes.Count;
        }

        int groupCount = 0; // �Ψӭp��h�� group
        int currentIndex = 0;
        int totalNotes = sourceNotes.Count;

        while (currentIndex < totalNotes)
        {
            List<OldNote> currentGroup = new List<OldNote>();
            OldNote firstNote = sourceNotes[currentIndex];
            currentGroup.Add(firstNote);

            // --- ���մ꺡 n �ӡA�P���ˬd�W������ ---
            int lookAheadCount = 1;
            while (lookAheadCount < n && (currentIndex + lookAheadCount) < totalNotes)
            {
                OldNote nextNote = sourceNotes[currentIndex + lookAheadCount];
                float timeDiff = nextNote.time - firstNote.time;
                if (timeDiff > beatThreshold_up) break;

                // Debug.Log("Diff: " + timeDiff);

                currentGroup.Add(nextNote);
                lookAheadCount++;
            }

            // --- ���� 3: �ˬd�O�_�L�u�ݭn��� ---
            bool shouldAddGroup = true;
            if (currentGroup.Count < n)
            {
                float groupDuration = currentGroup[currentGroup.Count - 1].time - currentGroup[0].time;
                // �p�G���� n �B�ɶ��t�ӵu -> ���
                if (groupDuration < beatThreshold_down)
                {
                    shouldAddGroup = false;
                }
            }

            // --- ��s���޻P���ηs�W�h ---
            if (shouldAddGroup)
            {
                // [�ק�] ���Υ[�J List�A�����p�ƾ� + 1
                groupCount++;

                // �w�]�G���L Group �ƶq + 1 (�R���᭱���@�� Note)
                int skipAmount = currentGroup.Count + 1;

                // �ˬd�G�U�@�� Note �O�_���o�@�ժ��̫�@�� Note �ӻ� (> 1��)
                int nextNoteIndex = currentIndex + currentGroup.Count;

                // �T�O�S���W�X�d��
                if (nextNoteIndex < totalNotes)
                {
                    OldNote lastNoteInGroup = currentGroup[currentGroup.Count - 1];
                    OldNote nextNoteToCheck = sourceNotes[nextNoteIndex];

                    float gap = nextNoteToCheck.time - lastNoteInGroup.time;

                    // �p�G���j�W�L 1 �� -> �אּ�u���R���v
                    // �]�N�O�u���L currentGroup.Count�A�O�d nextNoteToCheck ���U�@��
                    if (gap > beatThreshold_1s)
                    {
                        skipAmount = currentGroup.Count;
                    }
                }

                // ������ާ�s
                currentIndex += skipAmount;
            }
            else
            {
                // �p�G�o�ճQ���F�A���L�o�ժ� Note�A���ݭn�B�~�A�R���᭱�@��
                currentIndex += currentGroup.Count;
            }
        }
        // �^�ǭp��X�Ӫ��ƶq
        return groupCount;
    }
}
