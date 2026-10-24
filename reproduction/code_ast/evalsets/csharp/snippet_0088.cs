using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ProtoBuf;

namespace HoppoAlpha.DataLibrary.DataObject
{
    /// <summary>
    /// Configのパラメーターをシリアル化するためのクラス
    /// </summary>
    [ProtoContract]
    public class ConfigSerializeItem
    {
        /// <summary>
        /// JSONを表示するオプション
        /// </summary>
        [ProtoMember(1)]
        public bool ShowJson { get; set; }
        /// <summary>
        /// 戦闘ネタバレを表示するオプション
        /// </summary>
        [ProtoMember(2)]
        public bool ShowBattleInfo { get; set; }
        /// <summary>
        /// 最前面に表示するオプション
        /// </summary>
        [ProtoMember(3)]
        public bool ShowTopMost { get; set; }
        /// <summary>
        /// 【廃止】袖裏表示の非表示オプション
        /// </summary>
        [Obsolete]
        [ProtoMember(4)]
        public bool ShowSodeuraDisable { get; set; }
        /// <summary>
        /// Fiddlerの待ち受けポート番号
        /// </summary>
        [ProtoMember(5)]
        public int PortNumber { get; set; }
        /// <summary>
        /// スクリーンショットの保存ディレクトリ
        /// </summary>
        [ProtoMember(6)]
        public string ScreenshotDirectory { get; set; }
        /// <summary>
        /// プロキシのアドレス
        /// </summary>
        [ProtoMember(7)]
        public string ProxyAddress { get; set; }
        /// <summary>
        /// ブラウザの倍率
        /// </summary>
        [ProtoMember(8)]
        public int Ratio { get; set; }
        /// <summary>
        /// メインフォームの横サイズ
        /// </summary>
        [ProtoMember(9)]
        public int FormSizeX { get; set; }
        /// <summary>
        /// メインフォームの縦サイズ
        /// </summary>
        [ProtoMember(10)]
        public int FormSizeY { get; set; }
        /// <summary>
        /// 【廃止】キャラ数・装備数を表示しない
        /// </summary>
        [Obsolete]
        [ProtoMember(11)]//使用しない
        public bool DoNotShowShipSlotitemNum { get; set; }
        /// <summary>
        /// 【廃止】タイマー情報を表示しない
        /// </summary>
        [Obsolete]
        [ProtoMember(12)]//使用しない
        public bool DoNotShowMissionNdockTimer { get; set; }
        /// <summary>
        /// 受信専用モードのフラグ
        /// </summary>
        [ProtoMember(13)]
        public bool ListeningMode { get; set; }
        /// <summary>
        /// 艦娘タブの自動更新を無効にするかどうか
        /// </summary>
        [ProtoMember(14)]
        public bool TabKanmusuAutoRefreshDisable { get; set; }
        /// <summary>
        /// ドッキングレイアウトの固定
        /// </summary>
        [ProtoMember(15)]
        public bool DockFixing { get; set; }
        /// <summary>
        /// ログのコンバートで出力するディレクトリ
        /// </summary>
        [ProtoMember(16)]
        public string LogExportOutputDirectory { get; set; }
        /// <summary>
        /// ブラウザツールでのお気に入りアドレス
        /// </summary>
        [ProtoMember(17)]
        public string[] BrowserUrlFavorite { get; set; }
        /// <summary>
        /// フォームの開始位置のX座標
        /// </summary>
        [ProtoMember(18)]
        public int FormLocationX { get; set; }
        /// <summary>
        /// フォームの開始位置のY座標
        /// </summary>
        [ProtoMember(19)]
        public int FormLocationY { get; set; }
        /// <summary>
        /// タイトルバーに任務情報を表示しないオプション
        /// </summary>
        [ProtoMember(20)]
        public bool QuestNotDisplayToForm { get; set; }
        /// <summary>
        /// ドロップレコードに追加しないオプション
        /// </summary>
        [ProtoMember(21)]
        public bool DropRecordAddDisable { get; set; }
        /// <summary>
        /// 装備検索で検索する分類
        /// </summary>
        [ProtoMember(22)]
        public int SlotitemSearchMainIndex { get; set; }
        /// <summary>
        /// 装備検索で検索する種別
        /// </summary>
        [ProtoMember(23)]
        public int SlotitemSearchSubIndex { get; set; }
        /// <summary>
        /// 艦娘タブで読まれたクエリ番号のコレクション
        /// </summary>
        [ProtoMember(24)]
        public List<int> TabKanmusuQueryNumber { get; set; }
        /// <summary>
        /// S艦隊タブの艦隊番号
        /// </summary>
        [ProtoMember(25)]
        public int TabFleetShortFleetIndex { get; set; }
        /// <summary>
        /// 出撃報告書の統合モード
        /// </summary>
        [ProtoMember(26)]
        public int SortieReportIntegrateMode { get; set; }
        /// <summary>
        /// 出撃報告書の表示の前景色
        /// </summary>
        [ProtoMember(27)]
        public string SortieReportViewForeColor { get; set; }
        /// <summary>
        /// 出撃報告書の表示の背景色
        /// </summary>
        [ProtoMember(28)]
        public string SortieReportViewBackColor { get; set; }
        /// <summary>
        /// 戦闘詳細で表示を保持するか
        /// </summary>
        [ProtoMember(29)]
        public bool BattleDetailViewKeeping { get; set; }
        /// <summary>
        /// JSONを表示している場合にJSONファイルを保存を無効にするか
        /// </summary>
        [ProtoMember(30)]
        public bool LoggingDisableOnShowJson { get; set; }
        /// <summary>
        /// 謎のおまじない
        /// </summary>
        [ProtoMember(31)]
        public string NazonoOmajonai { get; set; }
        /// <summary>
        /// UI更新タイマーのインターバル（ms）
        /// </summary>
        [ProtoMember(32)]
        public int UIRefresTimerhInterval { get; set; }
        /// <summary>
        /// 出撃報告書の期間統合モード
        /// </summary>
        [ProtoMember(33)]
        public int SortieReportTermIntegrateMode { get; set; }
        /// <summary>
        /// 終了時の確認を表示しないか（デフォルトで表示される）
        /// </summary>
        [ProtoMember(34)]
        public bool OnClosingNotifyDisable { get; set; }
        /// <summary>
        /// プリセット情報の重複チェックを旗艦に適用するか
        /// </summary>
        [ProtoMember(35)]
        public bool PresetDuplicateCheckAppliesFlagship { get; set; }

        //50-(AlphaDashの仕様)
        /// <summary>
        /// 【廃止】マルチスレッドを無効にする
        /// </summary>
        [Obsolete]
        [ProtoMember(51)]
        public bool DisableMultiThreading { get; set; }
        /// <summary>
        /// 【廃止】ワーカースレッドの数
        /// </summary>
        [Obsolete]
        [ProtoMember(52)]
        public int ThreadPoolMin { get; set; }
        /// <summary>
        /// ブラウザの高画質モード
        /// </summary>
        [ProtoMember(53)]
        public bool HighQualityMode { get; set; }
        /// <summary>
        /// マップ情報でクリア済みのマップを表示
        /// </summary>
        [ProtoMember(54)]
        public bool MapInfoShowCleared { get; set; }
        /// <summary>
        /// 戦果予測の保存ディレクトリ
        /// </summary>
        [ProtoMember(55)]
        public string SenkaPredictOutputDirectory { get; set; }
        /// <summary>
        /// バケツライン
        /// </summary>
        [ProtoMember(56)]
        public double BucketHPRatio { get; set; }
        /// <summary>
        /// 選択中の索敵モデル
        /// </summary>
        [ProtoMember(57)]
        public SearchModel.Models SearchUsingModel { get; set; }

        //100～104は古い仕様なので使用しない
        /// <summary>
        /// 【廃止】艦娘リストのモード
        /// </summary>
        [Obsolete]
        [ProtoMember(100)]//使用しない
        public UnitMaskMode UnitListMaskMode { get; set; }
        /// <summary>
        /// 【廃止】艦娘リストで一定時間以上の入渠を除外計算
        /// </summary>
        [Obsolete]
        [ProtoMember(101)]//使用しない
        public int UnitListCalcThresholdHour { get; set; }
        /// <summary>
        /// 【廃止】艦娘リストで一定時間以上の除外フラグ
        /// </summary>
        [Obsolete]
        [ProtoMember(102)]//使用しない
        public bool UnitListRemoveOverThresholdFlag { get; set; }
        /// <summary>
        /// 【廃止】艦娘リストで小破を除外
        /// </summary>
        [Obsolete]
        [ProtoMember(103)]//使用しない
        public bool UnitListRemoveSmallDamageFlag { get; set; }
        /// <summary>
        /// 【廃止】艦娘リストで艦隊所属艦の除外
        /// </summary>
        [Obsolete]
        [ProtoMember(104)]//使用しない
        public bool UnitListDoNotShowFleetAssignShips { get; set; }

        /// <summary>
        /// 遠征のポップアップを表示しない
        /// </summary>
        [ProtoMember(200)]
        public bool NotShowNotifyBalloonMission { get; set; }
        /// <summary>
        /// 入渠のポップアップを表示しない
        /// </summary>
        [ProtoMember(201)]
        public bool NotShowNotifyBalloonNdock { get; set; }
        /// <summary>
        /// 遠征のSEを鳴らさない
        /// </summary>
        [ProtoMember(202)]
        public bool SoundMissionDisableFlag { get; set; }
        /// <summary>
        /// 【廃止】遠征のSEのモード
        /// </summary>
        [Obsolete]
        [ProtoMember(203)]
        public int SoundMissionMode { get; set; }
        /// <summary>
        /// 遠征のSEのファイル名
        /// </summary>
        [ProtoMember(204)]
        public string SoundMissionFileName { get; set; }
        /// <summary>
        /// 入渠のSEを鳴らさない
        /// </summary>
        [ProtoMember(205)]
        public bool SoundNdockDisableFlag { get; set; }
        /// <summary>
        /// 【廃止】入渠のSEをのモード
        /// </summary>
        [Obsolete]
        [ProtoMember(206)]
        public int SoundNdockMode { get; set; }
        /// <summary>
        /// 入渠のSEのファイル名
        /// </summary>
        [ProtoMember(207)]
        public string SoundNdockFileName { get; set;}
        /// <summary>
        /// 大破警告のSEを鳴らさない
        /// </summary>
        [ProtoMember(208)]
        public bool SoundDamageDisableFlag { get; set; }
        /// <summary>
        /// 大破警告のSEのファイル名
        /// </summary>
        [ProtoMember(210)]
        public string SoundDamageFileName { get; set; }
        /// <summary>
        /// SEの音量(0～100)
        /// </summary>
        [ProtoMember(211)]
        public int SoundVolume { get; set; }
        /// <summary>
        /// 大破進撃警告を鳴らさない
        /// </summary>
        [ProtoMember(212)]
        public string SoundDamageSortieFileName { get; set; }
        /// <summary>
        /// 大破進撃警告のSEのファイル名
        /// </summary>
        [ProtoMember(213)]
        public bool SoundDamageSortieDisableFlag { get; set; }
        /// <summary>
        /// SEをミュートにするか
        /// </summary>
        [ProtoMember(222)]
        public bool SoundIsMuted { get; set; }

        /// <summary>
        /// 艦これデータベースのユーザートークン
        /// </summary>
        [ProtoMember(300)]
        public string KancolleDbUserToken { get; set; }
        /// <summary>
        /// 艦これDBへの送信を停止する（デフォルトで送信する）
        /// </summary>
        [ProtoMember(301)]
        public bool KancolleDbPostDisable = false;
        /// <summary>
        /// 艦これDBへの送信をプロキシモード時も行う（デフォルトでは行わない）
        /// </summary>
        [ProtoMember(302)]
        public bool KancolleDbPostOnProxyMode = false;
        /// <summary>
        /// 艦これ検証DBへの送信を有効にする（デフォルトでは行わない）
        /// </summary>
        [ProtoMember(350)]
        public bool KancolleVerifyPostEnable = false;
        /// <summary>
        /// 艦これ検証DBのダイアログを表示しない（デフォルトでは表示する）
        /// </summary>
        [ProtoMember(351)]
        public bool KancolleVerifyNotifyDialogNotShow = false;
        /// <summary>
        /// KCVDB画面でのUIタイマー更新頻度
        /// </summary>
        [ProtoMember(352)]
        public int KancolleVerifyScreenRefreshTimerInterval { get; set; }

        /// <summary>
        /// 潜水マンのリスト
        /// </summary>
        [ProtoMember(400)]
        public Dictionary<int, string> RankingSubmarinerList { get; set; }
        /// <summary>
        /// 潜水マンのEO不足分
        /// </summary>
        [ProtoMember(401)]
        public int RankingSubmarinerEOHandicap { get; set; }
        /// <summary>
        /// 戦果タブでボーダーを表示する順位
        /// </summary>
        [ProtoMember(402)]
        public int[] TabSenkaBorderDisplay { get; set; }
    }
}