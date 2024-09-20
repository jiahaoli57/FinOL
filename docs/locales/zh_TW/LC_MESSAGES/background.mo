Þ          ä               ¬  h   ­  [     
   r    }  L        S  ú   _  D   Z  È    V   h	  2   ¿	     ò	  )   û	  #   %
  ®   I
  P   ø
     I  e   ]    Ã  a   Ì    .  Ô   H      ü   +  ~  (  N   §  H   ö     ?  ^  L  8   «     ä  â   ë  >   Î      W     2   ò     %  !   ,     N     g  F        H  O   U  ü   ¥  W   ¢    ú    ÿ      ê       :math:`m`-dimensional non-negative price relatives vector at the end of the :math:`t`-th trading period. :math:`m`-dimensional portfolio vector at the beginning of the :math:`t`-th trading period. Background Before the start of the :math:`t`-th trading period, using past historical information, we can construct a portfolio vector :math:`\mathbf{b}_t=(b_{t,1},\ldots,b_{t,m}) \in \mathbb{R}^m` to allocate our wealth among :math:`m` assets. Each element :math:`b_{t,i}` represents the proportion of wealth invested in asset :math:`i` at the beginning of the :math:`t`-th trading period. A portfolio clearly needs to satisfy the simplex constraint :math:`\mathbf{b}_t \in \Delta_m`, where :math:`b_{t,i} \ge 0` and :math:`\sum\nolimits_{i=1}^{m}{{{b}_{t,i}}}=1`. This constraint indicates the portfolio is fully self-financing without leverage or shorting. Daily return of the portfolio at the end of the :math:`t`-th trading period. Description Early online portfolio selection (OLPS) methods relied on a priori assumptions about market dynamics and mathematical optimization. With increased computing power and data, data-driven OLPS methods that directly learn from data have gained attention. Final wealth achieved at the end of the :math:`n`-th trading period. In the context of online portfolio selection, we analyze a financial market consisting :math:`m` assets observed during a specified time horizon comprising :math:`n` discrete trading periods (the term "periods" is defined flexibly). At the end of the :math:`t`-th trading period, we use a :math:`m`-dimensional non-negative price relatives vector :math:`\mathbf{x}_t=(x_{t,1},\ldots,x_{t,m}) \in \mathbb{R}_{+}^{m}` to represent the performance of the :math:`m` assets, where each element :math:`x_{t,i}` equals the close price of asset :math:`i` on the :math:`t`-th trading period divided by the close price of asset :math:`i` on the :math:`(t-1)`-th trading period, i.e., :math:`x_{t,i}={C_{t,i}}/{C_{t-1,i}}`. Initial portfolio vector, typically a uniform distribution :math:`(1/m, \ldots, 1/m)`. Initial value of the portfolio, normally set to 1. Notation Number of assets in the financial market. Number of discrete trading periods. Obviously, the goal of this task is to maximize the final portfolio wealth, which depends entirely on the portfolio vector generated by the data-driven method in each period. Price relative of asset :math:`i` at the end of the :math:`t`-th trading period. Problem Formulation Proportion of wealth invested in asset :math:`i` at the beginning of the :math:`t`-th trading period. Researchers often default to assuming newer methods are superior, but this is irresponsible without consistent test conditions. Financial time series are highly non-stationary, so the same method can exhibit drastically different performance on different datasets. Simplex constraint indicating the portfolio is fully self-financing without leverage or shorting. Therefore, at the end of the :math:`t`-th trading period, the daily return of the portfolio is defined as :math:`\mathbf{b}_{t}^{\top} \mathbf{x}_t = \sum\nolimits_{i=1}^{m} b_{t,i}x_{t,i}`. Based on this, the final wealth achieved at the end of the :math:`n`-th trading period is: To address this, we introduce ``FinOL``, a new finance benchmark platform designed for data-driven OLPS research. ``FinOL`` provides diverse financial datasets and extensive benchmark results for fair comparison. While various data-driven OLPS methods have shown promise, evaluating and comparing them remains challenging. Without standardized datasets and fair comparisons, it's unclear which method truly performs best in real-world environments, hindering progress in this field. where, without loss of generality, the initial value of the portfolio is normally set to 1, i.e., :math:`{S}_{0} = 1`; at the same time, the portfolio vector is initialized to a uniform distribution, i.e., :math:`{\mathbf{b}}_{1} = (1/m, \ldots, 1/m)`. Project-Id-Version: FinOL 0.1.29
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
 å¨äº¤ææ :math:`t` çµææï¼:math:`m` ç¶­çéè² ç¸å°å¹æ ¼åéã å¨äº¤ææ :math:`t` éå§æï¼:math:`m` ç¶­çæè³çµååéã åé¡èæ¯ å¨ç¬¬ :math:`t` åäº¤ææéå§ä¹åï¼å©ç¨éå»çæ­·å²è³è¨ï¼æåå¯ä»¥æ§å»ºä¸åæè³çµååé :math:`\mathbf{b}_t=(b_{t,1},\ldots,b_{t,m}) \in \mathbb{R}^m` ä»¥å¨ :math:`m` åè³ç¢ä¸­åéæåçè²¡å¯ãæ¯ååç´  :math:`b_{t,i}` ä»£è¡¨å¨ç¬¬ :math:`t` åäº¤ææéå§æï¼æè³æ¼è³ç¢ :math:`i` çè²¡å¯æ¯ä¾ãæè³çµåé¡¯ç¶éè¦æ»¿è¶³ç°¡å®ç´æï¼:math:`\mathbf{b}_t \in \Delta_m`ï¼å¶ä¸­å¶ä¸­ï¼:math:`b_{t,i} \ge 0` ä¸ :math:`\sum\nolimits_{i=1}^{m}{{{b}_{t,i}}}=1`ã éåç´æè¡¨ææè³çµåæ¯å®å¨èªèè³çï¼ä¸åè¨±æ§æ¡¿æåç©ºã äº¤ææ :math:`t` çµæææè³çµåçæ¥åå ±ã æè¿° æ©æçå¨ç·æè³çµåé¸æï¼OLPSï¼æ¹æ³ä¾è³´æ¼å°å¸å ´åæåæ¸å­¸åªåçåé©åè¨­ãé¨èè¨ç®è½ååæ¸æçå¢å ï¼ç´æ¥å¾æ¸æä¸­å­¸ç¿çæ¸æé©åå OLPS æ¹æ³å¼èµ·äºäººåçéæ³¨ã å¨ç¬¬ :math:`n` åäº¤ææçµææç²å¾çæçµè²¡å¯ã å¨ç·ä¸æè³çµåé¸æçèæ¯ä¸ï¼æååæäºä¸åç± :math:`m` ç¨®è³ç¢çµæçéèå¸å ´ï¼éäºè³ç¢å¨ç± :math:`n` åé¢æ£äº¤ææ (è¡èª "å¨æ"çå®ç¾©æ¯éæ´»ç) çµæçç¹å®æéæ®µå§é²è¡è§å¯ãå¨ç¬¬ :math:`t` åäº¤ææçµææï¼æåä½¿ç¨ä¸å :math:`m` ç¶­çéè² ç¸å°å¹æ ¼åé :math:`\mathbf{x}_t=(x_{t,1},\ldots,x_{t,m}) \in \mathbb{R}_{+}^{m}` ä¾è¡¨ç¤º :math:`m` åè³ç¢çè¡¨ç¾ï¼å¶ä¸­åç´  :math:`x_{t,i}` ç­æ¼ç¬¬ :math:`t` åäº¤ææè³ç¢ :math:`i` çæ¶å¸å¹é¤ä»¥ç¬¬ :math:`(t-1)` åäº¤ææè³ç¢ :math:`i` çæ¶å¸å¹ï¼å³ :math:`x_{t,i}={C_{t,i}}/{C_{t-1,i}}`ã åå§æè³çµååéï¼éå¸¸æ¯åå»åå¸çåé :math:`(1/m, \ldots, 1/m)`ã æè³çµåçåå§å¹å¼ï¼éå¸¸è¨­å®çº 1ã ç¬¦è éèå¸å ´ä¸çè³ç¢æ¸éã é¢æ£äº¤ææç¸½æ¸ã é¡¯ç¶ï¼éé ä»»åçç®æ¨æ¯æå¤§åæçµçæè³çµåè²¡å¯ï¼èéå®å¨åæ±ºæ¼æ¸æé©åæ³å¨æ¯åææçæçæè³çµååéã å¨äº¤ææ :math:`t` çµææï¼è³ç¢ :math:`i` çç¸å°å¹æ ¼ã åé¡å®ç¾© å¨äº¤ææ :math:`t` éå§æï¼æè³æ¼è³ç¢ :math:`i` çè²¡å¯æ¯ä¾ã ç ç©¶äººå¡å¾å¾æé»èªèªçºæ°çæ¹æ³æ´åªç§ï¼ä½å¦ææ²¡æçµ±ä¸çæ¸¬è©¦æ¢ä»¶ï¼éç¨®åæ³æ¯ä¸è´è´£ä»»çãéèæéåºåé«åº¦éç©©å®ï¼å æ­¤åä¸åæ¹æ³å¨ä¸åçæ¸æéä¸å¯è½æè¡¨ç¾åºæªç¶ä¸åçæ§è½ã å®ç´å½¢ç´æï¼è¡¨ææè³çµåæ¯å®å¨èªèè³çï¼ä¸åè¨±æ§æ¡¿æåç©ºã å æ­¤ï¼å¨ç¬¬ :math:`t` åäº¤ææçµææï¼æè³çµåçæ¯æ¥åå ±å®ç¾©çº :math:`\mathbf{b}_{t}^{\top}  \mathbf{x}_t = \sum\nolimits_{i=1}^{m} b_{t,i}x_{t,i}`ãæ®æ­¤ï¼æè³çµåå¨ç¬¬ :math:`n` åäº¤ææçµææå¯¦ç¾çæçµè²¡å¯çºï¼ çºäºè§£æ±ºéååé¡ï¼æåæ¨åºäº ``FinOL`` ââ ä¸åå°çºæ¸æé©åå OLPS ç ç©¶èè¨­è¨çæ°åéèåºæºå¹³å°ã ``FinOL`` æä¾äºå¤æ¨£åçéèæ¸æéï¼ä¸¦çµ¦åºäºå»£æ³çåºæºæ¸¬è©¦çµæï¼ä»¥ä¾¿é²è¡å¬å¹³æ¯è¼ã éç¶åç¨®æ¸æé©ååç OLPS æ¹æ³é½é¡¯ç¤ºåºäºä¸å®çåæ¯ï¼ä½è©ä¼°åæ¯è¼å®åä»ç¶æ¯ä¸åææ°ãç±æ¼ç¼ºä¹æ¨æºåçæ¸æéåå¬å¹³çæ¯è¼ï¼ç®åéä¸æ¸æ¥åªç¨®æ¹æ³å¨çå¯¦ç°å¢ä¸­çè¡¨ç¾æä½³ï¼éé»ç¤äºè©²é åçé²æ­¥ã å¶ä¸­ï¼å¨ä¸åªå¤±ä¸è¬æ§çåæä¸ï¼æè³çµåçåå§å¼éå¸¸è¨­å®çº 1ï¼å³ï¼:math:`{S}_{0} = 1`ï¼åæï¼æè³çµååéåå§åçºåå»åå¸çåéï¼å³ï¼:math:`{\mathbf{b}}_{1} = (1/m, \ldots, 1/m)`ã 