��    B      ,              <     =     V     d     |     �     �     �     �  5   �  [       w  �   �  J   ^  J   �  M   �  L   B  L   �  J   �  O   '	  K   w	  J   �	  E   
  6   T
     �
     �
      �
     �
  P   �
  m  >  E   �  9   �  0   ,  �   ]  :   A  0   |  ?   �  H   �  /   6  �   f  /   �     ,  �   M  �   �  Z   �  _     �   a  B     E   I  j   �  C   �  �   >  V   �  b     X   �  f   �  �   A  �   �     T  a   Y  N   �     
                      ~  "     �  
   �     �     �     �                0  )   7  J   a  �   �  �   s  =   
  =   H  @   �  ?   �  ?     =   G  B   �  >   �  9     6   A  $   x     �     �     �     �  9   �  k     <   z!  F   �!     �!  �   "  0   �"     �"  9   �"  0   8#  $   i#  l   �#  '   �#  �   #$  u   �$  �   Z%  ?    &  R   @&  �   �&  *   /'  <   Z'  W   �'  6   �'  {   &(  I   �(  O   �(  J   <)  Q   �)  Q   �)  Q   +*     }*  _   �*  P   �*     5+     B+     O+     _+     l+   A value between 0 and 1. API Reference Any non-negative float. Any non-negative integer. Any positive float. Any positive integer. Data Layer Configuration Description Determine whether to check for updates for ``FinOL``. Determine whether to download the ``FinOL`` datasets and benchmark results from the source. Determines whether to include an economic distillation analysis as part of the overall analysis. The economic distillation analysis aims to identify the most important economic features that influence portfolio performance, allowing for a more focused and interpretable model. Determines whether to include an interpretability analysis as part of the overall analysis. The interpretability analysis aims to provide insights into the features that drive the generation of the portfolios. Determines whether to include the :ref:`OHLCV_features` in the input data. Determines whether to include the :ref:`cycle_features` in the input data. Determines whether to include the :ref:`momentum_features` in the input data. Determines whether to include the :ref:`overlap_features` in the input data. Determines whether to include the :ref:`pattern_features` in the input data. Determines whether to include the :ref:`price_features` in the input data. Determines whether to include the :ref:`volatility_features` in the input data. Determines whether to include the :ref:`volume_features` in the input data. Determines whether to include the look-back window data in the input data. Determines whether to load the dataloader from the local data source. Determines whether to perform hyper-parameters tuning. Evaluation Layer Configuration Model Layer Configuration Optimization Layer Configuration Options Sets the seed for random number generation to ensure reproducibility of results. Specifies the algorithm to be used for hyper-parameters tuning. See `optuna.samplers <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html>`__ and `Which sampler should be used? <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used>`__ for more details. Specifies the batch size to use during model training and validation. Specifies the dataset to use (:ref:`supported_datasets`). Specifies the device to be used for computation. Specifies the feature distiller to be used in the economic distillation analysis. This parameter determines the specific method that will be used to identify the most important features from the original set of input variables. Specifies the language to use for plot labels and legends. Specifies the model parameters and their values. Specifies the name of the criterion to be used during training. Specifies the number of epochs after which to save the model checkpoint. Specifies the number of training epochs to run. Specifies the number of trials to perform during hyper-parameters tuning. This determines how many different sets of hyper-parameters will be tested. Specifies the optimizer to use during training. Specifies the proportion of the most important features to be retained after the economic distillation process. This parameter determines how many of the original features will be used in the economic distillation model, with the goal of creating a more interpretable and efficient model. Specifies the proportion of winner assets to be invested during the actual investment process. This parameter determines how many of the best-performing assets will be invested. Specifies the pruner to be used for hyper-parameters tuning. See `optuna.pruners <https://optuna.readthedocs.io/en/stable/reference/pruners.html>`__ for more details. Specifies the set of model hyper-parameters to be explored during hyper-parameters tuning. Specifies the step size at each iteration while moving toward a minimum/maximum of a criterion. Specifies the strength of the L2 regularization. Required only when the ``CRITERION_NAME`` is set to ``LogWealthL2Diversification`` or ``LogWealthL2Concentration``. Specifies the target variable for the economic distillation model. Specifies the type of data scaling method to apply to the input data. Specifies the type of model to be used. Each model corresponds to a different neural network architecture. Specifies the window size use for containing look-back window data. Specifies the wrapped pruner to be used for hyper-parameters tuning. Required only when the ``PRUNER_NAME`` is set to ``PatientPruner``. The :mod:`~finol.data_layer` module contains data layer related classes and functions. The :mod:`~finol.evaluation_layer` module contains evaluation layer related classes and functions. The :mod:`~finol.model_layer` module contains model layer related classes and functions. The :mod:`~finol.optimization_layer` module contains optimization layer related classes and functions. The keys in the dictionary correspond to the names of the model parameters, and the values correspond to the desired parameter values. The keys in the dictionary correspond to the names of the model parameters, and the values correspond to the range of the parameter values. Type ``CNN``, ``DNN``, ``RNN``, ``LSTM``, ``CNN``, ``Transformer``, ``LSRE-CAAN``, ``AlphaPortfolio``. ``en`` (English), ``zh_CN`` (Chinese Simple), ``zh_TW`` (Chinese Traditional). bool dict float int str Project-Id-Version: FinOL 0.1.29
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2024-08-13 01:01+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_TW
Language-Team: zh_TW <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.14.0
 介於 0 至 1 之間的值 API 文檔 任何非負浮點數。 任何非負整數。 任何正浮點數。 任何正整數。 數據層配置 描述 確定是否檢查 ``FinOL`` 的更新。 確定是否從源代碼下載 ``FinOL`` 數據集和基準測試結果。 確定是否將經濟蒸餾分析納入整體分析的一部分。經濟蒸餾分析旨在識別影響投資組合表現的最重要經濟特徵，從而使模型更具有針對性和可解釋性。 確定是否將可解釋性分析納入整體分析的一部分。可解釋性分析旨在提供對驅動投資組合生成的特徵的深入了解。 確定是否在輸入數據中包含 :ref:`OHLCV_features`。 確定是否在輸入數據中包含 :ref:`cycle_features`。 確定是否在輸入數據中包含 :ref:`momentum_features`。 確定是否在輸入數據中包含 :ref:`overlap_features`。 確定是否在輸入數據中包含 :ref:`pattern_features`。 確定是否在輸入數據中包含 :ref:`price_features`。 確定是否在輸入數據中包含 :ref:`volatility_features`。 確定是否在輸入數據中包含 :ref:`volume_features`。 確定是否在輸入數據中包含回溯窗口數據。 確定是否從本地數據源加載數據加載器。 確定是否執行超參數調優。 評估層配置 模型層配置 優化層配置 選項 設定隨機數生成的種子，確保結果可重現。 指定用於超參數調優的算法。有關更多詳細信息，請參見 `optuna.samplers <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html>`__  和 `我應該使用哪種取樣器？ <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used>`__。 指定在模型訓練和驗證期間使用的批次大小。 指定要使用的數據集（參考：:ref:`supported_datasets`）。 指定用於計算的裝置。 指定經濟蒸餾分析中使用的特征蒸餾器。該參數決定了用於從原始輸入變量集中識別最重要特徵的具體方法。 指定繪圖標籤和圖例所使用的語言。 指定模型參數及其值。 指定在訓練期間要使用的評估標準的名稱。 指定在多少周期後保存模型檢查點。 指定要執行的訓練周期數。 指定在超參數調優期間執行的試驗次數。這確定了要測試的不同超參數集的數量。 指定訓練期間使用的優化器。 指定經濟蒸餾過程後要保留的最重要特徵比例。此參數決定了經濟蒸餾模型中將使用多少原始特徵，目的是創建一個更易於解釋且更有效的模型。 指定實際投資過程中要投資的優勝資產比例。此參數決定了多少表現最佳資產將被投資。 指定用於超參數調優的剪枝器。有關更多詳細資訊，請參見 `optuna.pruners <https://optuna.readthedocs.io/en/stable/reference/pruners.html>`__。 指定在超參數調優期間要探索的模型超參數集。 指定在向評估標準的最小值/最大值移動時的每次迭代的步長。 指定 L2 正則化的強度。該參數僅當 ``CRITERION_NAME`` 設置為 ``LogWealthL2Diversification`` 或 ``LogWealthL2Concentration`` 時才需要。 指定經濟蒸餾模型的目標變數。 指定要應用於輸入數據的數據縮放方法類型。 指定要使用的模型類型。每種模型都對應於不同的神經網絡架構。 指定用於包含回溯窗口數據的窗口大小。 指定用於超參數調優的包裹剪枝器。該參數僅當 ``PRUNER_NAME`` 設定為 ``PatientPruner`` 時才需要。 :mod:`~finol.data_layer` 模組包含與數據層相關的類和函數。 :mod:`~finol.evaluation_layer` 模組包含與評估層相關的類和函數。 :mod:`~finol.model_layer` 模組包含與模型層相關的類和函數。 :mod:`~finol.optimization_layer` 模組包含與優化層相關的類和函數。 字典中的鍵對應於模型參數的名稱，值對應於所需的參數值。 字典中的鍵對應於模型參數的名稱，值對應於參數值的範圍。 類型 ``CNN``, ``DNN``, ``RNN``, ``LSTM``, ``CNN``, ``變換器``, ``LSRE-CAAN``, ``AlphaPortfolio``. ``en`` （英文）, ``zh_CN`` （簡體中文）, ``zh_TW`` （繁體中文）. 布爾類型 字典類型 浮點數類型 整數類型 字符串類型 