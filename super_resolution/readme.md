<h1>  White Paper - Super-résolution d’images par intelligence artificielle - Fiction ou Révolution ? </h1>

<h3> Un notebook self-contained pour tester différents algorithmes de Deep Learning </h3>

<p>Mickaël Tits - CETIC asbl - https://www.cetic.be/Mickael-Tits?lang=fr - 03/07/2019</p>


Suite au succès de notre premier article de blog sur le Deep Learning, dans un contexte artistique (https://www.cetic.be/Deep-Learning-et-styles-artistiques), nous avons décidé de diffuser une série d’articles pour mieux faire connaître les applications du Deep Learning dans le traitement multimédia à un large public en francophonie, et plus particulièrement en Wallonie. En parallèle, La [Deep Learning Academy](https://www.linkedin.com/groups/8614751/&sa=D&ust=1566390330218000), lancée conjointement par UCLouvain et UMONS, et récemment rejointe par le CETIC, a entamé une initiative pour le test et la diffusion d’outils technologiques utilisant du Deep Learning, regroupés actuellement sous le lien suivant: https://deep-learning-academy.github.io/


<p align="justify">C’est donc dans cette démarche commune que nous proposons un premier article blanc, accompagné d’un [notebook colab](https://colab.research.google.com/github/titsitits/Test_images_superresolution/blob/master/Super_resolution_comparison.ipynb) de démonstration, sur la super-résolution d’images.</p>


Dans cet article, nous parlerons plus particulièrement d’un ensemble techniques permettant d’améliorer la qualité d’une image grâce à une augmentation artificielle des pixels. Ces techniques s’appellent couramment “super-résolution” d’image. Les méthodes les plus performantes aujourd’hui sont toutes basées sur les réseaux de neurones profonds (Deep Neural Networks - DNN).

<h2> Un peu de théorie sur le Deep Learning</h2>

<span class="c1 c4">Un réseau de neurones profond est un algorithme qui permet de prédire une variable dépendante à partir de plusieurs variables prédictives. L’exemple le plus couramment utilisé pour expliquer cette notion de modèle prédictif est celui du prix des maisons. A partir d’un ensemble d’exemples de maisons (appelé jeu d’entraînement), dont on connaît le prix, ainsi que différentes variables telles que la surface, le nombre de pièces, l’âge, etc., on estime une fonction qui va caractériser le prix en fonction des autres variables connues:</span>


<p align="center">
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image1.png" width="800" />
</p>


<span>L’architecture d’un réseau de neurones permet de définir une fonction plus ou moins complexe, avec des paramètres (les poids des liens entre les neurones) qui vont devoir être choisis de manière à estimer au mieux cette fonction. Ce choix des paramètres se fait de manière à ce que la fonction donne un résultat le plus proche possible du prix réel pour toutes les maisons du jeu d’entraînement. Ce processus est effectué par un processus d’optimisation, et plus particulièrement d’une minimisation de l’erreur des prédictions sur toutes les maisons du jeu d’entraînement (appelée fonction de coût)</span> <span class="c28">(c’est-à-dire une minimisation de l’écart entre leur prix réel et le prix prédit à partir de leur surface et de leur âge par la fonction estimée)</span><span>. Cette optimisation se base généralement sur l’algorithme de descente de gradient</span> <sup>[[1]](#ftnt1)</sup><span class="c1 c4">. Cet algorithme permet de calculer la modification des paramètres (les poids des liens entre les neurones) qui va permettre de faire baisser le plus l’erreur de prédiction. Ce processus est appliqué de manière itérative par petits pas jusqu’à atteindre un minimum de l’erreur de prédiction sur toutes les maisons d’entraînement. En général, on teste ensuite la fonction obtenue (le réseau de neurone avec ses paramètres bien choisis) sur un nouvel ensemble de maisons qui n’ont pas été utilisées pour l’entraînement, pour vérifier si la fonction permet de réellement estimer le prix des maisons, et n’a pas juste retenu par coeur le prix des maisons du jeu d’entraînement. On parle de phase de test.</span>


<p align="center">
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image3.png" width="800" />
</p>


<span>Figure 1\. Lien entre un réseau de neurone artificiel et le cerveau. Le modèle est constitué d’un ensemble de noeuds (les neurones) dotés de plusieurs entrées sur lesquelles est appliquée une fonction généralement non-linéaire, et reliés entre eux par des connexions dont les poids s’adaptent grâce à un algorithme d’optimisation. De la même manière, le cerveau est constitué de neurones reliés entre eux par des synapses dont les courants électriques et les connexions s’adaptent continuellement, et plus particulièrement lors de l’apprentissage de tâches complexes comme apprendre à jouer d’un instrument de musique.</span><sup>[[2]](#ftnt2)</sup>


<span>Ce concept général d’entraînement automatique d’un algorithme à partir d’exemples et de la minimisation d’une fonction de coût est le fondement du machine learning, une branche importante de l’intelligence artificielle. Le Deep Learning est une sous-branche particulière du machine learning, basée sur un type spécifique d’algorithmes, à savoir les réseaux de neurones. Les réseaux de neurones sont une famille d’algorithmes permettant d’estimer des fonctions extrêmement complexes, et sont appelés ainsi car ils sont inspirés par la manière dont fonctionne le cerveau (voir Figure 1). En effet, l’apprentissage au niveau du cerveau se fait par un réarrangement perpétuel des connexions (les synapses) entre un grand nombre de petites unités appelés neurones, permettant à un animal d’apprendre progressivement n’importe quelle tâche grâce des exemples et de l’entraînement, permettant ainsi de reconnaître un chat ou chien, de comprendre des mots, d’apprendre à marcher, etc. Ce phénomène est communément connu sous le nom de plasticité cérébrale</span><sup>[[3]](#ftnt3)</sup><span class="c1 c4">.</span>

# <span class="c4 c12 c21">Application à la super-résolution d’images</span>

<span class="c1 c4">Le machine learning peut être appliqué à de très nombreuses disciplines, à condition de bien choisir le modèle qui va permettre d’estimer une fonction entre des variables d’entrées (prédictives) et une ou plusieurs variables de sortie (à prédire), et à condition d’avoir un grand nombre d’exemples (le jeu d’entraînement). L’état de l’art dans ce domaine, i.e. le Deep Learning, permet aujourd’hui d’estimer des fonctions extrêmement complexes impliquant des variables d’entrées et de sortie tout aussi complexes (à conditions d’avoir un très grand nombre d’exemples, allant de plusieurs dizaines de milliers, à plusieurs dizaines millions selon la tâche à apprendre).</span>

<span class="c1 c4"></span>

<span class="c1 c4">Cette révolution technologique a permis d’étendre le domaine de l’intelligence artificielle à de nombreux nouveaux domaines, rendant possible de nouvelles applications, qui jusqu’alors relevaient du domaine de la science fiction. Ainsi, il est dès lors possible, comme on le voyait à l’époque de manière incrédule dans certains épisodes de la fameuse série “Les Experts”, d’augmenter artificiellement la résolution d’une image pour améliorer sa qualité (cfr Figure 2).</span>

<span class="c1 c4"></span>


<p align="center">
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image10.png" width="800" />
</p>

<span>Figure 2\. Extrait des experts à Miami.</span> <span class="c14">[https://www.youtube.com/watch?v=IRBo5ZGcyVA](https://www.google.com/url?q=https://www.youtube.com/watch?v%3DIRBo5ZGcyVA&sa=D&ust=1566390330224000)</span>

<span class="c1 c4">Dans ce contexte (i.e. la super-résolution d’image), le but est de prédire une image de meilleure qualité (plus réaliste, et ayant plus de pixels) à partir d’une image d’entrée plus petite (voir Figure 2) :</span>


<p align="center">
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image2.png" width="800" />
</p>


<span class="c1 c4">Pour réaliser cette tâche, le jeu d’entraînement consiste donc en un ensemble de paires d’images identiques mais de résolutions différentes. Ce jeu peut être obtenu soit en prenant deux photos identiques avec deux appareils photos différents, ou plus simplement en diminuant artificiellement la taille d’une image pour en extraire une version de plus basse résolution.</span>

<span class="c1 c4"></span>

<span>Afin d’entraîner un modèle (pour calculer et minimiser une fonction de coût), il existe des mesures permettant d’évaluer objectivement la qualité de reconstruction d’une image, en la comparant avec l’image originale. Les mesures habituellement utilisées sont le rapport signal à bruit (“Peak Signal-to-Noise Ratio” - PSNR)</span> <sup>[[4]](#ftnt4)</sup><span>, et la similarité structurelle (“Structural Similarity - SSIM)</span> <sup>[[5]](#ftnt5)</sup><span class="c1 c4">.</span>

<span class="c1 c4"></span>


<p align="center">
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image5.png" width="800" />
</p>


<span class="c1 c4">Figure 2\. Super-résolution d’image par apprentissage profond - principe de base. L’algorithme est entraîné à générer, à partir d’une image de taille artificiellement réduite, une image de plus grande résolution la plus proche possible de l’image originale.</span>


Ces dernières années, de nouveaux travaux sur cette thématique sont régulièrement proposés, afin d’améliorer les techniques de super-résolution d’images. Ces améliorations se basent fréquemment sur la proposition d’une meilleure architecture de réseau de neurone (c’est-à-dire un modèle plus adapté à la fonction à estimer, i.e. la super-résolution d’image dans ce contexte), sur des jeux de données plus larges, plus adaptés, ou encore sur une manière plus pertinente d’évaluer la qualité d’une image reconstruite. Pour comparer les dernières avancées dans l’état de l’art, des benchmarks basés sur ces mesures et sur un ensemble d’images dédiées sont utilisés. Par exemple, on trouve ici un classement de différents travaux, mesurés en PSNR et en SSIM, sur un ensemble de 14 images de référence:  https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling


<p align="center">
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image13.png" width="800" />
</p>


<span class="c1 c4">Figure 3\. Réseaux antagonistes génératifs (Generative Adversarial Networks - GAN). Le générateur est entraîné à générer une image de grande taille la plus réaliste possible à partir de l’image d’entrée (de petite taille), alors que le discriminateur est entraîné à différencier une image réelle d’une image synthétisée par le générateur. Le premier a donc comme objectif de maximiser l’erreur du deuxième.</span>

<span class="c1 c4"></span>

<span>Certains travaux s’appuient sur un nouveau type d’architecture de DNN, particulièrement ingénieuse: les réseaux antagonistes génératifs (Generative Adversarial Networks - GAN, voir Figure 3)</span><sup>[[6]](#ftnt6)</sup><span class="c1 c4">. Un GAN consiste en la mise en compétition de deux modèles distincts: un modèle “génératif” et un modèle “discriminatif”. Le premier est entraîné à reconstruire une image de grande résolution la plus réaliste possible à partir de l’image de faible résolution. Le second est par contre entraîné à partir des images produites par le premier, à déterminer si cette image est réelle ou non. Un mécanisme de feedback permet alors au premier modèle de s’adapter pour tenter de convaincre le mieux possible le deuxième modèle que les images qu’il synthétise sont réelles. Plus précisément, le premier modèle est littéralement optimisé pour maximiser l’erreur du deuxième, en générant des images les plus réalistes possibles. On peut dire en quelque sorte que le premier modèle est entraîné à berner le deuxième, qui lui est entraîné à être de plus en plus perspicace (à ne pas se faire berner).</span>

<span class="c1 c4"></span>

<span>Quelque soit le modèle, l’amélioration d’une image reste cependant une notion en partie subjective puisqu’elle dépend de la perception d’un individu. En outre, la dégradation d’une image lors de son acquisition dépend également du capteur utilisé, et selon le type de dégradation du capteur, différents modèles peuvent s’avérer plus adaptés que d’autres</span> <span class="c16 c7">[[1]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/dG6f&sa=D&ust=1566390330230000)</span><span>. Pour tenter de répondre à cette subjectivité, une mesure basée sur l’opinion moyenne de différentes personnes (“Mean Opinion Score” - MOS) est parfois utilisée. En l’occurrence, les méthodes basées sur les GANs ont des résultats particulièrement bons selon cette mesure</span> <span class="c16 c7">[[2]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/iO8r&sa=D&ust=1566390330230000)</span><span>.</span>

<span class="c1 c4"></span>

<span>Ainsi, de nombreux travaux se réclament supérieurs aux autres, sous couvert d’un benchmark bien précis et selon une mesure bien précise à laquelle leur modèle est plus adapté. Par exemple, Yu et al. (2018)</span> <span class="c16 c7">[[3]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/QmCt&sa=D&ust=1566390330230000)</span><span> indiquent sur leur répertoire github</span><sup>[[7]](#ftnt7)</sup><span>qu’ils ont gagné la compétition NTIRE 2018</span> <span class="c16 c7">[[4]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/a4re&sa=D&ust=1566390330231000)</span><span>. Mais d’autre part, Haris et al. (2018)</span> <span class="c16 c7">[[5]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/Uq6k&sa=D&ust=1566390330231000)</span><span> indiquent sur leur propre répertoire github</span><sup>[[8]](#ftnt8)</sup><span>avoir remporté la même compétition, ainsi qu’une autre, la PIRM 2018</span> <span class="c16 c7">[[6]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/7Jfw&sa=D&ust=1566390330231000)</span><span>... Cette dernière aurait cependant été également remportée par Wang et al. (2019)</span> <span class="c16 c7">[[7]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/66jd&sa=D&ust=1566390330232000)</span><span> comme ils le revendiquent également sur leur répertoire github.</span><sup>[[9]](#ftnt9)</sup><span class="c1 c4"> Bien sûr, chaque équipe a en réalité remporté une discipline spécifique, de la même manière que Kevin Borlée peut remporter la course au 400m mais être dernier au 100m haie aux mêmes Jeux Olympiques.</span>

# <span>Un notebook pour tester et comparer les modèles</span>

<span class="c1 c4">Dans cette jungle de méthodes et de mesures, il est difficile de se retrouver et de choisir la méthode qui convient le mieux aux images à traiter. C’est pourquoi le mieux est de tester par soi-même ces modèles et de les comparer sur ses propres images. Dans ce contexte, nous vous avons confectionné un tutoriel complet tenant dans un notebook unique sur Google Colab, et accessible sur le lien suivant:</span>

https://colab.research.google.com/github/titsitits/Test_images_superresolution/blob/master/Super_resolution_comparison.ipynb


<span class="c1 c4">Google Colab est une plateforme gratuite permettant de faire tourner des algorithmes en python sur une machine hébergée chez Google, et surtout dotée d’une carte graphique (GPU) suffisamment puissance pour faire tourner des algorithmes de traitement d’image utilisant des modèles de Deep Learning (vous aurez en effet du mal à faire tourner ces programmes sur votre pc portable).</span>

<span class="c1 c4"></span>

<span class="c1 c4">Dans ce notebook, nous testons six algorithmes de super-résolution d’images:</span>

<span class="c1 c4"></span>

1.  <span>Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)</span> <span class="c7 c16">[[8]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/c6Tx&sa=D&ust=1566390330235000)</span>
2.  <span>Wide Activation for Efficient and Accurate Image Super-Resolution (WDSR)</span> <span class="c16 c7">[[3]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/QmCt&sa=D&ust=1566390330235000)</span>
3.  <span>Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)</span> <span class="c16 c7">[[2]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/iO8r&sa=D&ust=1566390330236000)</span>
4.  <span>Enhanced super-resolution generative adversarial networks (ESRGAN)</span> <span class="c16 c7">[[7]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/66jd&sa=D&ust=1566390330236000)</span>
5.  <span>Deep Back-Projection Networks For Super-Resolution (DBPN)</span> <span class="c16 c7">[[5]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/Uq6k&sa=D&ust=1566390330237000)</span>
6.  <span>Feedback Network for Image Super-Resolution (SRFBN)</span> <span class="c16 c7">[[9]](https://www.google.com/url?q=https://paperpile.com/c/OM3Y77/jgr9&sa=D&ust=1566390330237000)</span>

<span class="c1 c4"></span>

<span class="c1 c4">Les cellules du notebook peuvent se lancer à la suite, une par une, simplement en cliquant dessus puis en appuyant sur Maj+Enter (ou Shift+Enter). Vous pouvez aussi tout exécuter d’un coup (Exécution => Tout exécuter, ou Ctrl+F9). Le notebook permet, étape par étape, de télécharger des petites images de test, de télécharger tous les modèles déjà entraînés sur un très grand jeu d’images d’entraînement, de les utiliser avec les images de test et enfin d’afficher les résultats pour comparaison.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Remarque sur l’utilisation du notebook: il semble que les deux derniers modèles testés (DBPN et SRFBN) soient particulièrement lourd pour le GPU, et ne semblent pas libérer complètement la mémoire après usage. Si vous relancer plusieurs fois le notebook sans le réinitialiser, vous risquez donc d’avoir des messages d’erreur indiquant le manque de mémoire GPU. Dans ce cas, vous devrez réinitialiser l’environnement d’exécution (Exécution => réinitialiser tous les environnements d’exécution). Si vous voulez réaliser plusieurs tests sans réinitialiser à chaque fois l’environnement, il est donc conseillé de se limiter aux premiers algorithmes.</span>

# <span class="c4 c12 c21">Résultats</span>

<span class="c1 c4">Les deux dernières cellules du notebook permettent de visualiser les résultats des différents algorithmes. Les images testées ont été récupérées manuellement sur Google Images en appliquant le filtre sur les images autorisant la réutilisation et la modification, à des fins de démonstration. Vous pouvez bien sûr réaliser le test avec vos propres images.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Dans le style des Experts à Miami, nous avons tenté de zoomer de petites parties de photos, pour tenter de permettre la reconnaissance d’un visage, d’un logo, ou de lire l’heure sur une montre.</span>

<span class="c1 c4"></span>

<span class="c1 c4">L’analyse de ces résultats est purement visuelle et donc tout à fait subjective.</span>

<span class="c1 c4"></span>

<span>L’amélioration la plus flagrante (de mon point de vue subjectif) a été obtenue sur un zoom sur un portrait, permettant notamment de reproduire une image d’oeil plutôt réaliste (voir Figure 4). On peut également remarquer que le contour des yeux, le sourcil et les cheveux semblent réalistes. Le meilleur résultat a été obtenu (selon moi) avec ESRGAN, qui se trouve être justement le premier dans plusieurs classement de méthodes sur le site</span> <span class="c14">[paperswithcode](https://www.google.com/url?q=https://paperswithcode.com/task/image-super-resolution&sa=D&ust=1566390330239000)</span><span> (voir Figure 5)</span><span class="c1 c4">.</span>

<p align="center">
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image4.png" width="400" />
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image8.png" width="400" />

<span class="c1 c4">Figure 4\. Exemple de super-résolution d’image, obtenu avec ESRGAN.</span>
</p>

<span class="c1 c4"></span>

<span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 602.00px; height: 241.33px;">![](https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image7.png)</span>

<span>Figure 5\. Etat de l’art recensé sur paperswithcode.com (résupéré le 03/07/2019 -</span> <span class="c14">[https://paperswithcode.com/task/image-super-resolution](https://www.google.com/url?q=https://paperswithcode.com/task/image-super-resolution&sa=D&ust=1566390330241000)</span><span class="c1 c4">). SRGAN + Residual-in-Rseidual Dense Block (ESRGAN) se retrouve à la première place sur la plupart des benchmarks.</span>


<span>A l’inverse, de nombreux exemples moins concluants ont été obtenus avec chaque algorithmes, comme on peut le voir à la Figure 6\. L’ensemble des résultats peut être vu (ou regénéré) à partir du</span> <span class="c14">[notebook colab](https://www.google.com/url?q=https://colab.research.google.com/github/titsitits/Test_images_superresolution/blob/master/Super_resolution_comparison.ipynb&sa=D&ust=1566390330242000)</span><span class="c1 c4"> fourni avec cet article.</span>


<p align="center">
 <img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image11.png" width="400" />  
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image9.png" width="400" />
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image6.png" width="400" />  
<img src="https://titsitits.github.io/open-image-restoration/White%20paper%20-%20super-resolution%20(french)/Whitepapersuperresolution_fichiers/image12.png" width="400" />

Figure 6\. Autres exemples de super-résolution d’image avec ESRGAN.<sup>[[10]](#ftnt10)</sup>
L’algorithme semble ajouter du bruit sur certaines zones de l’image de manière parfois peu réaliste.
</p>

<span class="c1 c4"></span>

# <span class="c4 c12 c21">Verdict</span>

<span class="c1 c4">Les films/séries policières et de science fiction ont montré plusieurs fois le concept de super-résolution d’images, et son intérêt potentiel dans divers domaines, que ce soit pour améliorer des images issus de microscopes, de caméras de surveillance ou des photos historiques. Les développements récents en intelligence artificielle, et plus particulièrement dans les techniques d’apprentissage automatique appelées “Deep Learning”, permettent aujourd’hui de prétendre à ce genre d’exercice qui n’était à l’époque que pure fiction.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Les résultats obtenus semble prometteurs, et certains des exemples montrés semblent indiquer que les algorithmes sont capables de comprendre d’une certaine manière les images, tel que détecter un oeil, ou des cheveux, et de reproduire une texture crédible pour ces zones particulières.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Néanmoins, la plupart des algorithmes semble mieux fonctionner sur une image qui a été réduite artificiellement, que sur de réelles images à améliorer. Ceci est assez logique puisqu’en général, les algorithmes sont entraînés sur des images artificiellement réduites. Ces résultats expliquent pourquoi les recherches actuelles se focalisent sur l’estimation de la distorsion réelle d’image lors de sa captation.</span>

<span class="c1 c4"></span>

<span class="c1 c4">Les techniques actuelles sont encore loin de rivaliser avec les films de science fiction, mais montrent néanmoins un grand potentiel dans l’amélioration de la qualité d’image. Le réentraînement spécifique de certains algorithmes sur un jeu de données propre à un contexte (microscopie, astronomie, sécurité, restauration, etc.) permettrait sans doute d’obtenir des modèles spécialisés pour chaque contexte, donnant potentiellement de meilleurs résultats. Comme le dit l’adage: “Data is the new oil”.</span>

<span class="c1 c4"></span>

<span class="c1 c4"></span>

* * *

<span class="c1 c4"></span>

# <span class="c4 c12 c21">Références</span>

<span class="c1">1\.         </span><span class="c1 c7">[Chen C, Xiong Z, Tian X, Zha Z-J, Wu F. Camera Lens Super-Resolution [Internet]. 2019\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/dG6f&sa=D&ust=1566390330244000) </span><span class="c1 c7">[http://arxiv.org/abs/1904.03378](https://www.google.com/url?q=http://arxiv.org/abs/1904.03378&sa=D&ust=1566390330244000)</span>

<span class="c1">2\.         </span><span class="c1 c7">[Ledig C, Theis L, Huszar F, Caballero J, Cunningham A, Acosta A, et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network [Internet]. 2016\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/iO8r&sa=D&ust=1566390330245000) </span><span class="c1 c7">[http://arxiv.org/abs/1609.04802](https://www.google.com/url?q=http://arxiv.org/abs/1609.04802&sa=D&ust=1566390330245000)</span>

<span class="c1">3\.         </span><span class="c1 c7">[Yu J, Fan Y, Yang J, Xu N, Wang Z, Wang X, et al. Wide Activation for Efficient and Accurate Image Super-Resolution [Internet]. 2018\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/QmCt&sa=D&ust=1566390330245000) </span><span class="c1 c7">[http://arxiv.org/abs/1808.08718](https://www.google.com/url?q=http://arxiv.org/abs/1808.08718&sa=D&ust=1566390330246000)</span>

<span class="c1">4\.         </span><span class="c1 c7">[Timofte R, Gu S, Van Gool L, Zhang L, Yang M-H. NTIRE 2018 Challenge on Single Image Super-Resolution: Methods and Results [Internet]. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). 2018\. doi:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/a4re&sa=D&ust=1566390330246000)</span><span class="c1 c7">[10.1109/cvprw.2018.00130](https://www.google.com/url?q=http://dx.doi.org/10.1109/cvprw.2018.00130&sa=D&ust=1566390330246000)</span>

<span class="c1">5\.         </span><span class="c1 c7">[Haris M, Shakhnarovich G, Ukita N. Deep Back-Projection Networks For Super-Resolution [Internet]. 2018\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/Uq6k&sa=D&ust=1566390330247000) </span><span class="c1 c7">[http://arxiv.org/abs/1803.02735](https://www.google.com/url?q=http://arxiv.org/abs/1803.02735&sa=D&ust=1566390330247000)</span>

<span class="c1">6\.         </span><span class="c1 c7">[Blau Y, Mechrez R, Timofte R, Michaeli T, Zelnik-Manor L. The 2018 PIRM Challenge on Perceptual Image Super-resolution [Internet]. 2018\. Available:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/7Jfw&sa=D&ust=1566390330247000) </span><span class="c1 c7">[http://arxiv.org/abs/1809.07517](https://www.google.com/url?q=http://arxiv.org/abs/1809.07517&sa=D&ust=1566390330247000)</span>

<span class="c1">7\.         </span><span class="c1 c7">[Wang X, Yu K, Wu S, Gu J, Liu Y, Dong C, et al. ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks [Internet]. Lecture Notes in Computer Science. 2019\. pp. 63–79\. doi:](https://www.google.com/url?q=http://paperpile.com/b/OM3Y77/66jd&sa=D&ust=1566390330248000)</span><span class="c1 c7">[10.1007/978-3-030-11021-5_5](https://www.google.com/url?q=http://dx.doi.org/10.1007/978-3-030-11021-5_5&sa=D&ust=1566390330248000)</span>

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

[[3]](#ftnt_ref3)<span class="c3">Plasticité cérébrale:</span> <span class="c14 c3">[https://fr.wikipedia.org/wiki/Plasticit%C3%A9_neuronale](https://www.google.com/url?q=https://fr.wikipedia.org/wiki/Plasticit%25C3%25A9_neuronale&sa=D&ust=1566390330251000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[4]](#ftnt_ref4)<span class="c3">PSNR:</span> <span class="c14 c3">[https://fr.wikipedia.org/wiki/Peak_Signal_to_Noise_Ratio](https://www.google.com/url?q=https://fr.wikipedia.org/wiki/Peak_Signal_to_Noise_Ratio&sa=D&ust=1566390330250000)</span><span class="c4 c12 c3"> </span>

</div>

<div>

[[5]](#ftnt_ref5)<span class="c3">Similarité structurelle:</span> <span class="c14 c3">[https://fr.wikipedia.org/wiki/Structural_Similarity](https://www.google.com/url?q=https://fr.wikipedia.org/wiki/Structural_Similarity&sa=D&ust=1566390330250000)</span><span class="c4 c12 c3"> </span>

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

[[10]](#ftnt_ref10)<span class="c3"> </span><span>Source de la deuxième image:</span> <span class="c14">[https://pxhere.com/en/photo/1213414](https://www.google.com/url?q=https://pxhere.com/en/photo/1213414&sa=D&ust=1566390330253000)</span>

</div>
