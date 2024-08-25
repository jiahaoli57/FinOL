Þ    !      $              ,  E  -  ì   s     `     s  B    l   Ì  	   9     C    Þ  à   u     V	     d	     s	  f  	  
   ù
  #     0   (  ¹   Y            &   $  ,   K     x          ±  ¡   Ë  ¶   m  4   $     Y      ç  E       N  ~  ^  ú   Ý  Â   Ø          ±  %  Ä  T   ê     ?  p   X  Z  É  Ø   $     ý       !       :     L     Y  -   r        	   =     G  '   W  '     !   §  !   É  $   ë  {          .        J  !   å  ,       4   Before running the above commands, users can first configure some parameters through the config file to customize the usage according to their needs. For example, setting the device, selecting a dataset, adjusting the data pre-processing parameters, and choosing a model, etc. The specific configuration method is as follows: By using the ``FinOL`` GUI, users can quickly and easily configure, train, and evaluate financial models without the need to write complex code. The intuitive interface make the process more accessible and user-friendly for researchers. Command Line Usage Dynamic Window Layout For example, when the user selects different model architectures in the "Model Layer" panel, the configuration options will dynamically update to display the specific parameters for that model. This dynamic layout allows users to focus on configuring the model without the need to switch between different tabs or windows. Found at the bottom right, this area is dedicated to displaying internal outputs, such as benchmark results. GUI Usage In addition to the above functionalities, the ``FinOL`` GUI interface also features some unique interactive capabilities that enhance the user experience. In addition to the command line usage, ``FinOL`` also providesp a GUI interface that allows users to achieve the same functionality as the command line usage in a more intuitive and visual way. The GUI interace includes options for dataset selection, model configuration, training, and evaluation, allowing users to easily customize the parameters and run the exeriments without the need to write any code. Located on the left, it contains various buttons for actions such as customizing datasets, models, and criteria. It also includes options to load datasets, train models, evaluate models, quit the application, and restart it. Open in Colab Output Display Overall Framework of FinOL GUI Positioned at the top right, this section allows users to switch between different configuration layers, such as Data Layer, Model Layer, Optimization Layer, and Evaluation Layer. Each tab provides specific settings and options for configuring the respective layer, including device and dataset selection, feature inclusion, and data augmentation parameters. Quickstart Real-time Configuration File Update Real-time Configuration File Update of FinOL GUI Regardless of the approach, users can always maintain a consistent running environment and parameter settings, significantly enhancing the flexibility and maintainability of the system. Sidebar Tab View The Dynamic Window Layout of FinOL GUI The GUI is divided into three main sections: The Output Display of FinOL GUI The Sidebar of FinOL GUI The Tab View of FinOL GUI The ``FinOL`` GUI employs a dynamic layout design, where the corresponding configuration panels automatically change based on the user's selections and settings. The ``FinOL`` GUI not only provides a visual interface for configuration settings, but also automatically updates the config. json file in the root directory with the user's changes. This guide will help you get started with ``FinOL``. To lower the barriers for the research community, ``FinOL`` provides a complete data-training-testing suite with just three lines of command. Unique Features of the FinOL GUI Whenever the user modifies any parameter in the GUI, the configuration file is instantly updated to reflect the latest settings. This real-time read-write functionality of the configuration file provides a seamless workflow, allowing users to switch between the GUI and the command-line interface without any inconsistencies. |Open in Colab| Project-Id-Version: FinOL 0.1.29
Report-Msgid-Bugs-To: 
POT-Creation-Date: 2024-08-13 01:01+0800
PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE
Last-Translator: FULL NAME <EMAIL@ADDRESS>
Language: zh_CN
Language-Team: zh_CN <LL@li.org>
Plural-Forms: nplurals=1; plural=0;
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: 8bit
Generated-By: Babel 2.14.0
 å¨è¿è¡ä¸è¿°å½ä»¤ä¹åï¼ç¨æ·å¯ä»¥åéè¿éç½®æä»¶è®¾ç½®ä¸äºåæ°ï¼ä»¥æ ¹æ®èªå·±çéæ±è¿è¡å®å¶ãä¾å¦è®¾ç½®ä½¿ç¨çè®¾å¤ãéæ©æ°æ®éãè°æ´æ°æ®é¢å¤çåæ°ä»¥åéæ©æ¨¡åç­ãå·ä½çéç½®æ¹æ³å¦ä¸: éè¿ä½¿ç¨ ``FinOL`` å¾å½¢çé¢ï¼ç¨æ·å¯ä»¥å¿«éè½»æ¾å°éç½®ï¼è®­ç»åè¯ä¼°éèæ¨¡åï¼èæ éç¼åå¤æçä»£ç ãç´è§ççé¢ä½¿ç ç©¶äººåæ´å®¹æä½¿ç¨åæä½ã å½ä»¤è¡ä½¿ç¨æ¹æ³ å¨æçªå£å¸å± ä¾å¦ï¼å½ç¨æ·å¨"æ¨¡åå±"ä¸­éæ©ä¸åçæ¨¡åæ¶ææ¶ï¼ç¸åºçéç½®éé¡¹æ¡å°±ä¼èªå¨ååï¼å±ç¤ºåºè¯¥æ¨¡åç¹æçåæ°è®¾ç½®é¡¹ãè¿ç§å¨æååçå¸å±ï¼å¯ä»¥è®©ç¨æ·æ´ä¸æ³¨å°éç½®æ¨¡åï¼èä¸éè¦å¨ä¸åçéé¡¹å¡æçªå£ä¹é´æ¥ååæ¢ã è¯¥åºåä½äºå³ä¸æ¹ï¼ä¸é¨ç¨äºæ¾ç¤ºåé¨è¾åºï¼å¦åºåæµè¯ç»æã å¾å½¢çé¢ä½¿ç¨æ¹æ³ é¤äºä¸è¿°åè½å¤ï¼``FinOL`` å¾å½¢çé¢è¿å·æä¸äºç¬ç¹çäº¤äºåè½ï¼å¯ä»¥å¢å¼ºç¨æ·ä½éªã é¤äºå½ä»¤è¡ä½¿ç¨æ¹å¼å¤ï¼``FinOL`` è¿æä¾äºä¸ä¸ªå¾å½¢çé¢ï¼GUIï¼ï¼ä½¿ç¨æ·å¯ä»¥ä»¥æ´ç´è§åå¯è§åçæ¹å¼å®ç°ä¸å½ä»¤è¡ä½¿ç¨ç¸åçåè½ãè¿ä¸ªå¾å½¢çé¢åæ¬æ°æ®ééæ©ãæ¨¡åéç½®ãè®­ç»åè¯ä¼°ç­éé¡¹ï¼å¯ä»¥è®©ç¨æ·å¯ä»¥è½»æ¾èªå®ä¹åæ°å¹¶è¿è¡å®éªï¼æ éç¼åä»»ä½ä»£ç ã è¯¥åºåä½äºå·¦ä¾§ï¼åå«ç¨äºèªå®ä¹æ°æ®éãæ¨¡ååè¯å¤æ åç­æä½çåç§æé®ãå®è¿åæ¬å è½½æ°æ®éãè®­ç»æ¨¡åãè¯ä¼°æ¨¡åãéåºå¾å½¢çé¢åéæ°å¾å½¢çé¢çéé¡¹ã Open in Colab è¾åºæ¾ç¤º FinOL å¾å½¢çé¢çæ»ä½æ¡æ¶ è¯¥åºåä½äºå³ä¸æ¹ï¼åè®¸ç¨æ·å¨ä¸åçéç½®å±ä¹é´åæ¢ï¼ä¾å¦æ°æ®å±ãæ¨¡åå±ãä¼åå±åè¯ä¼°å±ãæ¯ä¸ªéé¡¹å¡é½æä¾äºç¨äºéç½®åèªå±çç¹å®è®¾ç½®åéé¡¹ï¼åæ¬è®¾å¤åæ°æ®ééæ©ãç¹å¾åå«åæ°æ®å¢å¼ºåæ°ã å¿«éå¥é¨ å®æ¶éç½®æä»¶æ´æ° FinOL å¾å½¢çé¢çå®æ¶éç½®æä»¶æ´æ° æ è®ºæ¯éè¿å¾å½¢çé¢è¿æ¯å½ä»¤è¡ï¼ç¨æ·é½å¯ä»¥è·å¾å®å¨ä¸è´çè¿è¡ç¯å¢ååæ°è®¾ç½®ï¼è¿å¤§å¤§æé«äºçµæ´»æ§åå¯ç»´æ¤æ§ã ä¾§è¾¹æ  éé¡¹å¡è§å¾ FinOL å¾å½¢çé¢çå¨æçªå£å¸å± å¾å½¢çé¢åä¸ºä¸ä¸ªä¸»è¦é¨åï¼ FinOL å¾å½¢çé¢çè¾åºæ¾ç¤º FinOL å¾å½¢çé¢ççä¾§è¾¹æ  FinOL å¾å½¢çé¢çéé¡¹å¡è§å¾ FinOL çå¾å½¢çé¢éç¨äºå¨æå¸å±è®¾è®¡ï¼æ ¹æ®ç¨æ·çéæ©åè®¾ç½®ï¼ç¸åºçéç½®çªå£ä¼å¨æååã FinOL çå¾å½¢çé¢ä¸ä»æä¾ç´è§çéç½®è®¾ç½®ï¼èä¸å¯ä»¥å®æ¶å°ç¨æ·çè®¾ç½®æ´æ°å°æ ¹ç®å½ä¸ç config.json æä»¶ä¸­ã æ¬æåå°å¸®å©æ¨å¼å§ä½¿ç¨ ``FinOL``ã ä¸ºäºéä½ç ç©¶äººåçä½¿ç¨é¨æ§ï¼``FinOL`` æä¾äºä»é 3 è¡å½ä»¤å°±è½å®ææ°æ®å è½½ãæ¨¡åè®­ç»åæµè¯çä¸ç«å¼è§£å³æ¹æ¡ã FinOL å¾å½¢çé¢çç¬ç¹ä¹å¤ æ¯å½ç¨æ·å¨å¾å½¢çé¢ä¸­ä¿®æ¹ä»»ä½åæ°ï¼éç½®æä»¶é½ä¼è¢«èªå¨æ´æ°ï¼ä»¥åæ ææ°çè®¾ç½®ãè¿ç§å®æ¶è¯»åéç½®æä»¶çåè½ï¼ä¸ºç¨æ·æä¾äºä¸ç§æ ç¼çå·¥ä½æµï¼åè®¸ç¨æ·å¨å¾å½¢ç¨æ·çé¢åå½ä»¤è¡çé¢ä¹é´åæ¢ï¼èä¸ä¼åºç°ä»»ä½ä¸ä¸è´ã |Open in Colab| 