<span class="c4 c12 c20">White Paper - Super-r�solution d�images par intelligence artificielle - Fiction ou R�volution ?</span>

<span class="c4 c31">Un notebook self-contained pour tester diff�rents algorithmes de Deep Learning</span>

<span>Micka�l Tits - CETIC asbl -</span> <span class="c14">[https://www.cetic.be/Mickael-Tits?lang=fr](https://www.google.com/url?q=https://www.cetic.be/Mickael-Tits?lang%3Dfr&sa=D&ust=1566390330216000)</span><span> - 03/07/2019</span>

<span class="c1 c4"></span>

<span class="c1 c4"></span>

<span class="c1 c4"></span>

<span>Suite au succ�s de notre premier article de blog sur le Deep Learning, dans un contexte artistique (</span> <span class="c14">[https://www.cetic.be/Deep-Learning-et-styles-artistiques](https://www.google.com/url?q=https://www.cetic.be/Deep-Learning-et-styles-artistiques&sa=D&ust=1566390330218000)</span><span>), nous avons d�cid� de diffuser une s�rie d�articles pour mieux faire conna�tre les applications du Deep Learning dans le traitement multim�dia � un large public en francophonie, et plus particuli�rement en Wallonie. En parall�le, La</span> <span class="c14">[Deep Learning Academy](https://www.google.com/url?q=https://www.linkedin.com/groups/8614751/&sa=D&ust=1566390330218000)</span><span class="c1 c4">, lanc�e conjointement par UCLouvain et UMONS, et r�cemment rejointe par le CETIC, a entam� une initiative pour le test et la diffusion d�outils technologiques utilisant du Deep Learning, regroup�s actuellement sous le lien suivant:</span>

<span class="c14">[https://deep-learning-academy.github.io/](https://www.google.com/url?q=https://deep-learning-academy.github.io/&sa=D&ust=1566390330219000)</span><span class="c1 c4"> </span>

<span class="c1 c4"></span>

<span>C�est donc dans cette d�marche commune que nous proposons un premier article blanc, accompagn� d�un</span> <span class="c14">[notebook colab](https://www.google.com/url?q=https://colab.research.google.com/github/titsitits/Test_images_superresolution/blob/master/Super_resolution_comparison.ipynb&sa=D&ust=1566390330219000)</span><span class="c1 c4"> de d�monstration, sur la super-r�solution d�images.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Dans cet article, nous parlerons plus particuli�rement d�un ensemble techniques permettant d�am�liorer la qualit� d�une image gr�ce � une augmentation artificielle des pixels. Ces techniques s�appellent couramment �super-r�solution� d�image. Les m�thodes les plus performantes aujourd�hui sont toutes bas�es sur les r�seaux de neurones profonds (Deep Neural Networks - DNN).</span>

# <span class="c4 c12 c21">Un peu de th�orie sur le Deep Learning</span>

<span class="c1 c4">Un r�seau de neurones profond est un algorithme qui permet de pr�dire une variable d�pendante � partir de plusieurs variables pr�dictives. L�exemple le plus couramment utilis� pour expliquer cette notion de mod�le pr�dictif est celui du prix des maisons. A partir d�un ensemble d�exemples de maisons (appel� jeu d�entra�nement), dont on conna�t le prix, ainsi que diff�rentes variables telles que la surface, le nombre de pi�ces, l��ge, etc., on estime une fonction qui va caract�riser le prix en fonction des autres variables connues:</span>

<span class="c1 c4"></span>

![](Whitepapersuperresolution_fichiers/image1.png)

<span class="c1 c4"></span>

<span>L�architecture d�un r�seau de neurones permet de d�finir une fonction plus ou moins complexe, avec des param�tres (les poids des liens entre les neurones) qui vont devoir �tre choisis de mani�re � estimer au mieux cette fonction. Ce choix des param�tres se fait de mani�re � ce que la fonction donne un r�sultat le plus proche possible du prix r�el pour toutes les maisons du jeu d�entra�nement. Ce processus est effectu� par un processus d�optimisation, et plus particuli�rement d�une minimisation de l�erreur des pr�dictions sur toutes les maisons du jeu d�entra�nement (appel�e fonction de co�t)</span> <span class="c28">(c�est-�-dire une minimisation de l��cart entre leur prix r�el et le prix pr�dit � partir de leur surface et de leur �ge par la fonction estim�e)</span><span>. Cette optimisation se base g�n�ralement sur l�algorithme de descente de gradient</span> <sup>[[1]](#ftnt1)</sup><span class="c1 c4">. Cet algorithme permet de calculer la modification des param�tres (les poids des liens entre les neurones) qui va permettre de faire baisser le plus l�erreur de pr�diction. Ce processus est appliqu� de mani�re it�rative par petits pas jusqu�� atteindre un minimum de l�erreur de pr�diction sur toutes les maisons d�entra�nement. En g�n�ral, on teste ensuite la fonction obtenue (le r�seau de neurone avec ses param�tres bien choisis) sur un nouvel ensemble de maisons qui n�ont pas �t� utilis�es pour l�entra�nement, pour v�rifier si la fonction permet de r�ellement estimer le prix des maisons, et n�a pas juste retenu par coeur le prix des maisons du jeu d�entra�nement. On parle de phase de test.</span>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 540.96px; height: 323.50px;">![](Whitepapersuperresolution_fichiers/image3.png)</span>

<span>Figure 1\. Lien entre un r�seau de neurone artificiel et le cerveau. Le mod�le est constitu� d�un ensemble de noeuds (les neurones) dot�s de plusieurs entr�es sur lesquelles est appliqu�e une fonction g�n�ralement non-lin�aire, et reli�s entre eux par des connexions dont les poids s�adaptent gr�ce � un algorithme d�optimisation. De la m�me mani�re, le cerveau est constitu� de neurones reli�s entre eux par des synapses dont les courants �lectriques et les connexions s�adaptent continuellement, et plus particuli�rement lors de l�apprentissage de t�ches complexes comme apprendre � jouer d�un instrument de musique.</span><sup>[[2]](#ftnt2)</sup>

<span class="c1 c4"></span>

<span>Ce concept g�n�ral d�entra�nement automatique d�un algorithme � partir d�exemples et de la minimisation d�une fonction de co�t est le fondement du machine learning, une branche importante de l�intelligence artificielle. Le Deep Learning est une sous-branche particuli�re du machine learning, bas�e sur un type sp�cifique d�algorithmes, � savoir les r�seaux de neurones. Les r�seaux de neurones sont une famille d�algorithmes permettant d�estimer des fonctions extr�mement complexes, et sont appel�s ainsi car ils sont inspir�s par la mani�re dont fonctionne le cerveau (voir Figure 1). En effet, l�apprentissage au niveau du cerveau se fait par un r�arrangement perp�tuel des connexions (les synapses) entre un grand nombre de petites unit�s appel�s neurones, permettant � un animal d�apprendre progressivement n�importe quelle t�che gr�ce des exemples et de l�entra�nement, permettant ainsi de reconna�tre un chat ou chien, de comprendre des mots, d�apprendre � marcher, etc. Ce ph�nom�ne est commun�ment connu sous le nom de plasticit� c�r�brale</span><sup>[[3]](#ftnt3)</sup><span class="c1 c4">.</span>

# <span class="c4 c12 c21">Application � la super-r�solution d�images</span>

<span class="c1 c4">Le machine learning peut �tre appliqu� � de tr�s nombreuses disciplines, � condition de bien choisir le mod�le qui va permettre d�estimer une fonction entre des variables d�entr�es (pr�dictives) et une ou plusieurs variables de sortie (� pr�dire), et � condition d�avoir un grand nombre d�exemples (le jeu d�entra�nement). L��tat de l�art dans ce domaine, i.e. le Deep Learning, permet aujourd�hui d�estimer des fonctions extr�mement complexes impliquant des variables d�entr�es et de sortie tout aussi complexes (� conditions d�avoir un tr�s grand nombre d�exemples, allant de plusieurs dizaines de milliers, � plusieurs dizaines millions selon la t�che � apprendre).</span>

<span class="c1 c4"></span>

<span class="c1 c4">Cette r�volution technologique a permis d��tendre le domaine de l�intelligence artificielle � de nombreux nouveaux domaines, rendant possible de nouvelles applications, qui jusqu�alors relevaient du domaine de la science fiction. Ainsi, il est d�s lors possible, comme on le voyait � l��poque de mani�re incr�dule dans certains �pisodes de la fameuse s�rie �Les Experts�, d�augmenter artificiellement la r�solution d�une image pour am�liorer sa qualit� (cfr Figure 2).</span>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 546.00px; height: 365.75px;">![](Whitepapersuperresolution_fichiers/image10.png)</span>

<span>Figure 2\. Extrait des experts � Miami.</span> <span class="c14">[https://www.youtube.com/watch?v=IRBo5ZGcyVA](https://www.google.com/url?q=https://www.youtube.com/watch?v%3DIRBo5ZGcyVA&sa=D&ust=1566390330224000)</span>

<span class="c1 c4"></span>

<span class="c1 c4">Dans ce contexte (i.e. la super-r�solution d�image), le but est de pr�dire une image de meilleure qualit� (plus r�aliste, et ayant plus de pixels) � partir d�une image d�entr�e plus petite (voir Figure 2) :</span>

![](Whitepapersuperresolution_fichiers/image2.png)

<span class="c1 c4"></span>

<span class="c1 c4">Pour r�aliser cette t�che, le jeu d�entra�nement consiste donc en un ensemble de paires d�images identiques mais de r�solutions diff�rentes. Ce jeu peut �tre obtenu soit en prenant deux photos identiques avec deux appareils photos diff�rents, ou plus simplement en diminuant artificiellement la taille d�une image pour en extraire une version de plus basse r�solution.</span>

<span class="c1 c4"></span>

<span>Afin d�entra�ner un mod�le (pour calculer et minimiser une fonction de co�t), il existe des mesures permettant d��valuer objectivement la qualit� de reconstruction d�une image, en la comparant avec l�image originale. Les mesures habituellement utilis�es sont le rapport signal � bruit (�Peak Signal-to-Noise Ratio� - PSNR)</span> <sup>[[4]](#ftnt4)</sup><span>, et la similarit� structurelle (�Structural Similarity - SSIM)</span> <sup>[[5]](#ftnt5)</sup><span class="c1 c4">.</span>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 366.50px; height: 250.83px;">![](Whitepapersuperresolution_fichiers/image5.png)</span>

<span class="c1 c4">Figure 2\. Super-r�solution d�image par apprentissage profond - principe de base. L�algorithme est entra�n� � g�n�rer, � partir d�une image de taille artificiellement r�duite, une image de plus grande r�solution la plus proche possible de l�image originale.</span>

<span class="c1 c4"></span>

<span class="c1 c4"></span>

<span>Ces derni�res ann�es, de nouveaux travaux sur cette th�matique sont r�guli�rement propos�s, afin d�am�liorer les techniques de super-r�solution d�images. Ces am�liorations se basent fr�quemment sur la proposition d�une meilleure architecture de r�seau de neurone (c�est-�-dire un mod�le plus adapt� � la fonction � estimer, i.e. la super-r�solution d�image dans ce contexte), sur des jeux de donn�es plus larges, plus adapt�s, ou encore sur une mani�re plus pertinente d��valuer la qualit� d�une image reconstruite. Pour comparer les derni�res avanc�es dans l��tat de l�art, des benchmarks bas�s sur ces mesures et sur un ensemble d�images d�di�es sont utilis�s. Par exemple, on trouve ici un classement de diff�rents travaux, mesur�s en PSNR et en SSIM, sur un ensemble de 14 images de r�f�rence:</span> <span class="c14">[https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling](https://www.google.com/url?q=https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling&sa=D&ust=1566390330228000)</span>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 602.00px; height: 250.67px;">![](Whitepapersuperresolution_fichiers/image13.png)</span>

<span class="c1 c4">Figure 3\. R�seaux antagonistes g�n�ratifs (Generative Adversarial Networks - GAN). Le g�n�rateur est entra�n� � g�n�rer une image de grande taille la plus r�aliste possible � partir de l�image d�entr�e (de petite taille), alors que le discriminateur est entra�n� � diff�rencier une image r�elle d�une image synth�tis�e par le g�n�rateur. Le premier a donc comme objectif de maximiser l�erreur du deuxi�me.</span>

<span class="c1 c4"></span>

<span>Certains travaux s�appuient sur un nouveau type d�architecture de DNN, particuli�rement ing�nieuse: les r�seaux antagonistes g�n�ratifs (Generative Adversarial Networks - GAN, voir Figure 3)</span><sup>[[6]](#ftnt6)</sup><span class="c1 c4">. Un GAN consiste en la mise en comp�tition de deux mod�les distincts: un mod�le �g�n�ratif� et un mod�le �discriminatif�. Le premier est entra�n� � reconstruire une image de grande r�solution la plus r�aliste possible � partir de l�image de faible r�solution. Le second est par contre entra�n� � partir des images produites par le premier, � d�terminer si cette image est r�elle ou non. Un m�canisme de feedback permet alors au premier mod�le de s�adapter pour tenter de convaincre le mieux possible le deuxi�me mod�le que les images qu�il synth�tise sont r�elles. Plus pr�cis�ment, le premier mod�le est litt�ralement optimis� pour maximiser l�erreur du deuxi�me, en g�n�rant des images les plus r�alistes possibles. On peut dire en quelque sorte que le premier mod�le est entra�n� � berner le deuxi�me, qui lui est entra�n� � �tre de plus en plus perspicace (� ne pas se faire berner).</span>

<span class="c1 c4"></span>

<span>Quelque soit le mod�le, l�am�lioration d�une image reste cependant une notion en partie subjective puisqu�elle d�pend de la perception d�un individu. En outre, la d�gradation d�une image lors de son acquisition d�pend �galement du capteur utilis�, et selon le type de d�gradation du capteur, diff�rents mod�les peuvent s�av�rer plus adapt�s que d�autres</span> <span class="c16 c7">[[1]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/dG6f&sa=D&ust=1566390330230000)</span><span>. Pour tenter de r�pondre � cette subjectivit�, une mesure bas�e sur l�opinion moyenne de diff�rentes personnes (�Mean Opinion Score� - MOS) est parfois utilis�e. En l�occurrence, les m�thodes bas�es sur les GANs ont des r�sultats particuli�rement bons selon cette mesure</span> <span class="c16 c7">[[2]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/iO8r&sa=D&ust=1566390330230000)</span><span>.</span>

<span class="c1 c4"></span>

<span>Ainsi, de nombreux travaux se r�clament sup�rieurs aux autres, sous couvert d�un benchmark bien pr�cis et selon une mesure bien pr�cise � laquelle leur mod�le est plus adapt�. Par exemple, Yu et al. (2018)</span> <span class="c16 c7">[[3]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/QmCt&sa=D&ust=1566390330230000)</span><span> indiquent sur leur r�pertoire github</span><sup>[[7]](#ftnt7)</sup><span>qu�ils ont gagn� la comp�tition NTIRE 2018</span> <span class="c16 c7">[[4]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/a4re&sa=D&ust=1566390330231000)</span><span>. Mais d�autre part, Haris et al. (2018)</span> <span class="c16 c7">[[5]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/Uq6k&sa=D&ust=1566390330231000)</span><span> indiquent sur leur propre r�pertoire github</span><sup>[[8]](#ftnt8)</sup><span>avoir remport� la m�me comp�tition, ainsi qu�une autre, la PIRM 2018</span> <span class="c16 c7">[[6]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/7Jfw&sa=D&ust=1566390330231000)</span><span>... Cette derni�re aurait cependant �t� �galement remport�e par Wang et al. (2019)</span> <span class="c16 c7">[[7]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/66jd&sa=D&ust=1566390330232000)</span><span> comme ils le revendiquent �galement sur leur r�pertoire github.</span><sup>[[9]](#ftnt9)</sup><span class="c1 c4"> Bien s�r, chaque �quipe a en r�alit� remport� une discipline sp�cifique, de la m�me mani�re que Kevin Borl�e peut remporter la course au 400m mais �tre dernier au 100m haie aux m�mes Jeux Olympiques.</span>

# <span>Un notebook pour tester et comparer les mod�les</span>

<span class="c1 c4">Dans cette jungle de m�thodes et de mesures, il est difficile de se retrouver et de choisir la m�thode qui convient le mieux aux images � traiter. C�est pourquoi le mieux est de tester par soi-m�me ces mod�les et de les comparer sur ses propres images. Dans ce contexte, nous vous avons confectionn� un tutoriel complet tenant dans un notebook unique sur Google Colab, et accessible sur le lien suivant:</span>

<span class="c14">[https://colab.research.google.com/github/titsitits/Test_images_superresolution/blob/master/Super_resolution_comparison.ipynb](https://www.google.com/url?q=https://colab.research.google.com/github/titsitits/Test_images_superresolution/blob/master/Super_resolution_comparison.ipynb&sa=D&ust=1566390330234000)</span>

<span class="c1 c4"> </span>

<span class="c1 c4">Google Colab est une plateforme gratuite permettant de faire tourner des algorithmes en python sur une machine h�berg�e chez Google, et surtout dot�e d�une carte graphique (GPU) suffisamment puissance pour faire tourner des algorithmes de traitement d�image utilisant des mod�les de Deep Learning (vous aurez en effet du mal � faire tourner ces programmes sur votre pc portable).</span>

<span class="c1 c4"></span>

<span class="c1 c4">Dans ce notebook, nous testons six algorithmes de super-r�solution d�images:</span>

<span class="c1 c4"></span>

1.  <span>Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)</span> <span class="c7 c16">[[8]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/c6Tx&sa=D&ust=1566390330235000)</span>
2.  <span>Wide Activation for Efficient and Accurate Image Super-Resolution (WDSR)</span> <span class="c16 c7">[[3]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/QmCt&sa=D&ust=1566390330235000)</span>
3.  <span>Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)</span> <span class="c16 c7">[[2]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/iO8r&sa=D&ust=1566390330236000)</span>
4.  <span>Enhanced super-resolution generative adversarial networks (ESRGAN)</span> <span class="c16 c7">[[7]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/66jd&sa=D&ust=1566390330236000)</span>
5.  <span>Deep Back-Projection Networks For Super-Resolution (DBPN)</span> <span class="c16 c7">[[5]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/Uq6k&sa=D&ust=1566390330237000)</span>
6.  <span>Feedback Network for Image Super-Resolution (SRFBN)</span> <span class="c16 c7">[[9]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/jgr9&sa=D&ust=1566390330237000)</span>

<span class="c1 c4"></span>

<span class="c1 c4">Les cellules du notebook peuvent se lancer � la suite, une par une, simplement en cliquant dessus puis en appuyant sur Maj+Enter (ou Shift+Enter). Vous pouvez aussi tout ex�cuter d�un coup (Ex�cution => Tout ex�cuter, ou Ctrl+F9). Le notebook permet, �tape par �tape, de t�l�charger des petites images de test, de t�l�charger tous les mod�les d�j� entra�n�s sur un tr�s grand jeu d�images d�entra�nement, de les utiliser avec les images de test et enfin d�afficher les r�sultats pour comparaison.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Remarque sur l�utilisation du notebook: il semble que les deux derniers mod�les test�s (DBPN et SRFBN) soient particuli�rement lourd pour le GPU, et ne semblent pas lib�rer compl�tement la m�moire apr�s usage. Si vous relancer plusieurs fois le notebook sans le r�initialiser, vous risquez donc d�avoir des messages d�erreur indiquant le manque de m�moire GPU. Dans ce cas, vous devrez r�initialiser l�environnement d�ex�cution (Ex�cution => r�initialiser tous les environnements d�ex�cution). Si vous voulez r�aliser plusieurs tests sans r�initialiser � chaque fois l�environnement, il est donc conseill� de se limiter aux premiers algorithmes.</span>

# <span class="c4 c12 c21">R�sultats</span>

<span class="c1 c4">Les deux derni�res cellules du notebook permettent de visualiser les r�sultats des diff�rents algorithmes. Les images test�es ont �t� r�cup�r�es manuellement sur Google Images en appliquant le filtre sur les images autorisant la r�utilisation et la modification, � des fins de d�monstration. Vous pouvez bien s�r r�aliser le test avec vos propres images.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Dans le style des Experts � Miami, nous avons tent� de zoomer de petites parties de photos, pour tenter de permettre la reconnaissance d�un visage, d�un logo, ou de lire l�heure sur une montre.</span>

<span class="c1 c4"></span>

<span class="c1 c4">L�analyse de ces r�sultats est purement visuelle et donc tout � fait subjective.</span>

<span class="c1 c4"></span>

<span>L�am�lioration la plus flagrante (de mon point de vue subjectif) a �t� obtenue sur un zoom sur un portrait, permettant notamment de reproduire une image d�oeil plut�t r�aliste (voir Figure 4). On peut �galement remarquer que le contour des yeux, le sourcil et les cheveux semblent r�alistes. Le meilleur r�sultat a �t� obtenu (selon moi) avec ESRGAN, qui se trouve �tre justement le premier dans plusieurs classement de m�thodes sur le site</span> <span class="c14">[paperswithcode](https://www.google.com/url?q=https://paperswithcode.com/task/image-super-resolution&sa=D&ust=1566390330239000)</span><span> (voir Figure 5)</span><span class="c1 c4">.</span>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 248.61px; height: 231.50px;">![](Whitepapersuperresolution_fichiers/image4.png)</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 248.69px; height: 231.50px;">![](Whitepapersuperresolution_fichiers/image8.png)</span>

<span class="c1 c4">Figure 4\. Exemple de super-r�solution d�image, obtenu avec ESRGAN.</span>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 602.00px; height: 241.33px;">![](Whitepapersuperresolution_fichiers/image7.png)</span>

<span>Figure 5\. Etat de l�art recens� sur paperswithcode.com (r�sup�r� le 03/07/2019 -</span> <span class="c14">[https://paperswithcode.com/task/image-super-resolution](https://www.google.com/url?q=https://paperswithcode.com/task/image-super-resolution&sa=D&ust=1566390330241000)</span><span class="c1 c4">). SRGAN + Residual-in-Rseidual Dense Block (ESRGAN) se retrouve � la premi�re place sur la plupart des benchmarks.</span>

<span class="c1 c4"></span>

<span>A l�inverse, de nombreux exemples moins concluants ont �t� obtenus avec chaque algorithmes, comme on peut le voir � la Figure 6\. L�ensemble des r�sultats peut �tre vu (ou reg�n�r�) � partir du</span> <span class="c14">[notebook colab](https://www.google.com/url?q=https://colab.research.google.com/github/titsitits/Test_images_superresolution/blob/master/Super_resolution_comparison.ipynb&sa=D&ust=1566390330242000)</span><span class="c1 c4"> fourni avec cet article.</span>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 283.50px; height: 203.87px;">![](Whitepapersuperresolution_fichiers/image11.png)</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 288.50px; height: 203.77px;">![](Whitepapersuperresolution_fichiers/image9.png)</span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 287.00px; height: 249.74px;">![](Whitepapersuperresolution_fichiers/image6.png)</span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 287.87px; height: 250.50px;">![](Whitepapersuperresolution_fichiers/image12.png)</span>

<span>Figure 6\. Autres exemples de super-r�solution d�image avec ESRGAN.</span><sup>[[10]](#ftnt10)</sup><span class="c1 c4"> L�algorithme semble ajouter du bruit sur certaines zones de l�image de mani�re parfois peu r�aliste.</span>

<span class="c1 c4"></span>

# <span class="c4 c12 c21">Verdict</span>

<span class="c1 c4">Les films/s�ries polici�res et de science fiction ont montr� plusieurs fois le concept de super-r�solution d�images, et son int�r�t potentiel dans divers domaines, que ce soit pour am�liorer des images issus de microscopes, de cam�ras de surveillance ou des photos historiques. Les d�veloppements r�cents en intelligence artificielle, et plus particuli�rement dans les techniques d�apprentissage automatique appel�es �Deep Learning�, permettent aujourd�hui de pr�tendre � ce genre d�exercice qui n��tait � l��poque que pure fiction.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Les r�sultats obtenus semble prometteurs, et certains des exemples montr�s semblent indiquer que les algorithmes sont capables de comprendre d�une certaine mani�re les images, tel que d�tecter un oeil, ou des cheveux, et de reproduire une texture cr�dible pour ces zones particuli�res.</span>

<span class="c1 c4"></span>

<span class="c1 c4">N�anmoins, la plupart des algorithmes semble mieux fonctionner sur une image qui a �t� r�duite artificiellement, que sur de r�elles images � am�liorer. Ceci est assez logique puisqu�en g�n�ral, les algorithmes sont entra�n�s sur des images artificiellement r�duites. Ces r�sultats expliquent pourquoi les recherches actuelles se focalisent sur l�estimation de la distorsion r�elle d�image lors de sa captation.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Les techniques actuelles sont encore loin de rivaliser avec les films de science fiction, mais montrent n�anmoins un grand potentiel dans l�am�lioration de la qualit� d�image. Le r�entra�nement sp�cifique de certains algorithmes sur un jeu de donn�es propre � un contexte (microscopie, astronomie, s�curit�, restauration, etc.) permettrait sans doute d�obtenir des mod�les sp�cialis�s pour chaque contexte, donnant potentiellement de meilleurs r�sultats. Comme le dit l�adage: �Data is the new oil�.</span>

<span class="c1 c4"></span>

<span class="c1 c4"></span>

* * *

<span class="c1 c4"></span>

# <span class="c4 c12 c21">R�f�rences</span>

<span class="c1">1\.         </span><span class="c1 c7">[Chen C, Xiong Z, Tian X, Zha Z-J, Wu F. Camera Lens Super-Resolution [Internet]. 2019\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/dG6f&sa=D&ust=1566390330244000) </span><span class="c1 c7">[http://arxiv.org/abs/1904.03378](https://www.google.com/url?q=http://arxiv.org/abs/1904.03378&sa=D&ust=1566390330244000)</span>

<span class="c1">2\.         </span><span class="c1 c7">[Ledig C, Theis L, Huszar F, Caballero J, Cunningham A, Acosta A, et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network [Internet]. 2016\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/iO8r&sa=D&ust=1566390330245000) </span><span class="c1 c7">[http://arxiv.org/abs/1609.04802](https://www.google.com/url?q=http://arxiv.org/abs/1609.04802&sa=D&ust=1566390330245000)</span>

<span class="c1">3\.         </span><span class="c1 c7">[Yu J, Fan Y, Yang J, Xu N, Wang Z, Wang X, et al. Wide Activation for Efficient and Accurate Image Super-Resolution [Internet]. 2018\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/QmCt&sa=D&ust=1566390330245000) </span><span class="c1 c7">[http://arxiv.org/abs/1808.08718](https://www.google.com/url?q=http://arxiv.org/abs/1808.08718&sa=D&ust=1566390330246000)</span>

<span class="c1">4\.         </span><span class="c1 c7">[Timofte R, Gu S, Van Gool L, Zhang L, Yang M-H. NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results [Internet]. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2018\. doi:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/a4re&sa=D&ust=1566390330246000)</span><span class="c1 c7">[10.1109/cvprw.2018.00130](https://www.google.com/url?q=http://dx.doi.org/10.1109/cvprw.2018.00130&sa=D&ust=1566390330246000)</span>

<span class="c1">5\.         </span><span class="c1 c7">[Haris M, Shakhnarovich G, Ukita N. Deep Back-Projection Networks For Super-Resolution [Internet]. 2018\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/Uq6k&sa=D&ust=1566390330247000) </span><span class="c1 c7">[http://arxiv.org/abs/1803.02735](https://www.google.com/url?q=http://arxiv.org/abs/1803.02735&sa=D&ust=1566390330247000)</span>

<span class="c1">6\.         </span><span class="c1 c7">[Blau Y, Mechrez R, Timofte R, Michaeli T, Zelnik-Manor L. The 2018 PIRM Challenge on Perceptual Image Super-resolution [Internet]. 2018\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/7Jfw&sa=D&ust=1566390330247000) </span><span class="c1 c7">[http://arxiv.org/abs/1809.07517](https://www.google.com/url?q=http://arxiv.org/abs/1809.07517&sa=D&ust=1566390330247000)</span>

<span class="c1">7\.         </span><span class="c1 c7">[Wang X, Yu K, Wu S, Gu J, Liu Y, Dong C, et al. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks [Internet]. Lecture Notes in Computer Science. 2019\. pp. 63�79\. doi:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/66jd&sa=D&ust=1566390330248000)</span><span class="c1 c7">[10.1007/978-3-030-11021-5_5](https://www.google.com/url?q=http://dx.doi.org/10.1007/978-3-030-11021-5_5&sa=D&ust=1566390330248000)</span>

<span class="c1">8\.         </span><span class="c1 c7">[Lim B, Son S, Kim H, Nah S, Lee KM. Enhanced Deep Residual Networks for Single Image Super-Resolution [Internet]. 2017\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/c6Tx&sa=D&ust=1566390330248000) </span><span class="c1 c7">[http://arxiv.org/abs/1707.02921](https://www.google.com/url?q=http://arxiv.org/abs/1707.02921&sa=D&ust=1566390330249000)</span>

<span class="c1">9\.         </span><span class="c1 c7">[Li Z, Yang J, Liu Z, Yang X, Jeon G, Wu W. Feedback Network for Image Super-Resolution [Internet]. 2019\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/jgr9&sa=D&ust=1566390330249000) </span><span class="c1 c7">[http://arxiv.org/abs/1903.09814](https://www.google.com/url?q=http://arxiv.org/abs/1903.09814&sa=D&ust=1566390330249000)</span>

<span class="c1 c4"></span>

<span class="c1 c4"></span>

* * *

<div>

[[1]](#ftnt_ref1)<span class="c4 c12 c3"> Descente du gradient: https://fr.wikipedia.org/wiki/Algorithme_du_gradient</span>

</div>

<div>

[[2]](#ftnt_ref2)<span class="c3">Sources:</span> <span class="c14 c3">[https://cs231n.github.io/neural-networks-1/](https://www.google.com/url?q=https://cs231n.github.io/neural-networks-1/&sa=D&ust=1566390330254000)</span><span class="c3">,</span> <span class="c14 c3">[https://svgsilh.com/image/155655.html](https://www.google.com/url?q=https://svgsilh.com/image/155655.html&sa=D&ust=1566390330254000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[3]](#ftnt_ref3)<span class="c3">Plasticit� c�r�brale:</span> <span class="c14 c3">[https://fr.wikipedia.org/wiki/Plasticit%C3%A9_neuronale](https://www.google.com/url?q=https://fr.wikipedia.org/wiki/Plasticit%25C3%25A9_neuronale&sa=D&ust=1566390330251000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[4]](#ftnt_ref4)<span class="c3">PSNR:</span> <span class="c14 c3">[https://fr.wikipedia.org/wiki/Peak_Signal_to_Noise_Ratio](https://www.google.com/url?q=https://fr.wikipedia.org/wiki/Peak_Signal_to_Noise_Ratio&sa=D&ust=1566390330250000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[5]](#ftnt_ref5)<span class="c3">Similarit� structurelle:</span> <span class="c14 c3">[https://fr.wikipedia.org/wiki/Structural_Similarity](https://www.google.com/url?q=https://fr.wikipedia.org/wiki/Structural_Similarity&sa=D&ust=1566390330250000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[6]](#ftnt_ref6)<span class="c3">Generative Adversarial Network:</span> <span class="c14 c3">[https://fr.wikipedia.org/wiki/R%C3%A9seaux_antagonistes_g%C3%A9n%C3%A9ratifs](https://www.google.com/url?q=https://fr.wikipedia.org/wiki/R%25C3%25A9seaux_antagonistes_g%25C3%25A9n%25C3%25A9ratifs&sa=D&ust=1566390330251000)</span><span class="c4 c3 c12"> </span>

</div>

<div>

[[7]](#ftnt_ref7)<span class="c3"> </span><span class="c14 c3">[https://github.com/krasserm/super-resolution](https://www.google.com/url?q=https://github.com/krasserm/super-resolution&sa=D&ust=1566390330252000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[8]](#ftnt_ref8)<span class="c3"> </span><span class="c14 c3">[https://github.com/alterzero/DBPN-Pytorch](https://www.google.com/url?q=https://github.com/alterzero/DBPN-Pytorch&sa=D&ust=1566390330252000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[9]](#ftnt_ref9)<span class="c3"> </span><span class="c14 c3">[https://github.com/xinntao/ESRGAN](https://www.google.com/url?q=https://github.com/xinntao/ESRGAN&sa=D&ust=1566390330253000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[10]](#ftnt_ref10)<span class="c3"> </span><span>Source de la deuxi�me image:</span> <span class="c14">[https://pxhere.com/en/photo/1213414](https://www.google.com/url?q=https://pxhere.com/en/photo/1213414&sa=D&ust=1566390330253000)</span>

</div>