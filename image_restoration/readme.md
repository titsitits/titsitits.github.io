# Restauration d'image par réseaux de neurones profonds

##### Cas d'application : un ouvrage en hommage aux sportifs belges, nivellois

Mickaël Tits - CETIC - 05/11/2019



## Un pipeline de techniques de restauration d'image

### Des techniques diverses et variées

L'état de l'art de la restauration d'image se divise actuellement en différentes techniques distinctes, permettant de corriger un type particulier d'artéfact indésirable présent dans une image, ou d'inférer artificiellement de l'information supplémentaire à l'image (telle que des couleurs, ou des détails). Dans le cadre du présent projet de recherche, nous nous sommes intéressés, jusqu'à présent, à quatre techniques en particulier:

1. La suppression de bruit gaussien
2. La suppression de rayures (bruit de bande - stripe noise)
3. La colorisation
4. La super-résolution

En pratique, une image de mauvaise qualité présentera différents types de défauts, et à des degrés divers. Une image historique sera souvent limitée à une échelle de gris, une faible résolution, des artéfacts dus à la détérioration d'un support physique ( tel que du papier), et à la numérisation de celui-ci. L'application individuelle des techniques susmentionnées n'aura alors qu'une portée limitée sur les possibilités de restauration de cette image. 

### Une mise en série compliquée

Dans ce projet, nous avons étudié la mise en série de ces différentes techniques, de manière à obtenir une restauration holistique d'une image. Cette mise en série n'est en réalité pas un problème trivial, car l'application d'une technique peut fortement impacter le résultat de la technique subséquente. Dans certains cas, elle peut améliorer le résultat: l'application de la super-résolution sur une image bruitée risque parfois d'augmenter les détails d'un artéfact indésirable, telle qu'une rayure, de la neige ou tout autre type de bruit. Si tel est le cas, l'application préalable d'une technique de réduction du bruit rendra plus efficace la super-résolution. 

L'impact des techniques précédant la colorisation d'une image est particulièrement important également : la colorisation par réseaux de neurones profonds se base en grande partie sur l'analyse sémantique du contenu d'une image. L'algorithme apprend dans une certaine mesure à reconnaître un objet, grâce à sa forme et sa texture, lui permettant ensuite de déterminer quelle est sa couleur la plus probable. Le bruit d'une image peut rendre difficile l'analyse sémantique, et en particulier l'analyse des textures des objets. 

A l'inverse, l'état de l'art de ces techniques est encore imparfait, et l'application de certaines techniques peut parfois engendrer une perte significative d'information, ou la création de nouveaux artéfacts indésirables. C'est notamment le cas de la réduction de bruit gaussien, qui a souvent tendance à trop "lisser" les images, c'est-à-dire à éliminer une partie de leur texture. Ainsi par exemple, la texture poreuse d'une surface de béton, poussiéreuse d'une terre battue, ou herbeuse d'un gazon peuvent devenir identiquement lisses après l'application imparfaite de cette technique. Dans ce cas, la colorisation d'un jardin ou d'un mur de briques peuvent résulter en une surface grise sans texture claire. Dès lors, l'ajustement minutieux de la séquence et des des paramètres d'un pipeline de techniques de restauration d'image devient important. Malheureusement, il est difficile, voire impossible, d'identifier un pipeline d'opérations idéale, car les résultats peuvent varier selon les caractéristiques et les défauts d'une image.

### Des outils hétérogènes et incompatibles

La littérature informatique se distingue souvent par sa vaste hétérogénéité. en effet, il possible d'implémenter un même algorithme avec de nombreux outils différents, souvent incompatibles entre eux. Différents langages de programmation, différents frameworks ou librairies, ou encore différentes versions d'une librairies peuvent être utilisées, rendant difficile l'utilisation concomitante de techniques dont les implémentations sont incompatibles entre elles. 

Dans le présent contexte, les algorithmes de Deep Learning sont le plus souvent implémentés en Python. Les librairies de Deep Learning les plus connues sont en effet Tensorflow, Keras et Pytorch, toutes trois utilisées en Python. Néanmoins, pour des raisons sans doute partiellement historiques, mais également de part la nature du traitement d'image, nécessitant souvent des processus d'algèbre matricielle, le langage Matlab est également souvent utilisé, particulièrement pour les techniques de réduction de bruit. C'est notamment le cas d'un algorithme actuellement premier dans certains classements de référence concernant la réduction de bruit : [ Multi-level Wavelet-CNN for Image Restoration - MWCNN (CVPR, 2018)](https://paperswithcode.com/paper/multi-level-wavelet-cnn-for-image-restoration#code)

Etant donné le côté fermé d'un langage comme Matlab (programme payant, closed-source), certains de ces algorithmes sont réécrit en Python. C'est notamment le cas d'un des premiers algorithmes de réduction de bruit utilisant les réseaux de neurones profonds : [ Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising - DnCNN (TIP, 2017)](https://github.com/cszn/DnCNN)

D'autres outils sont parfois utilisés pour optimiser les performances computationnelles, permettant ainsi un traitement plus rapide des images. 

### Des performances hétérogènes et peu réalistes

Chaque implémentation d'une technique est testée et validée différemment par son développeur. En général, les images ayant obtenus les meilleurs résultats sont utilisées pour présenter le travail. Afin de rendre plus objective la comparaison des performances des algorithmes, divers classements, basés sur des mesures standards et sur des jeux de données standards sont utilisés. Cependant, ceux-ci se limitent également à un ensemble restreint d'images d'un type particulier. On retrouve par exemple des classements sur les jeux de données [URBAN100](https://paperswithcode.com/sota/image-super-resolution-on-urban100-4x) (100 images de paysages urbains), et [Manga109](https://paperswithcode.com/sota/image-super-resolution-on-manga109-4x) (109 images d'animation). 

Le site paperswithcode.com référence de nombreux classements pour chacune des techniques évoquées plus haut. Pour la super-résolution par exemple, six classements sont actuellement référencés:

![image-20191105164649118](./image-20191105164649118.png) 

​							Classements en super-résolution d'image - paperwithcode.com (05/11/2019)



Le classement comparant le plus grand nombre d'algorithmes est actuellement le "Set5 - 4x upscaling", comparant les résultats sur seulement cinq images spécifiques. Cependant, les performances sur un ensemble de cinq images ne garantissent absolument pas l'efficacité sur d'autres images aux caractéristiques potentiellement très différentes. De plus, des contextes d'application typiques nécessitant de la restauration d'image (tel que des images prises par des capteurs low-cost, par des smartphones, de images nocturnes de vidéosurveillance, ou encore images historiques) ne sont pris en compte dans aucun de ces classements.

![image-20191105164930156](./image-20191105164930156.png)

​								Classement "Set5 - 4x upscaling" - paperwithcode.com (05/11/2019)

Pour terminer, bien que la qualité d'une image est un critère subjectif, des mesures objectives doivent être utilisées non-seulement pour entraîner les algorithmes (par apprentissage profond), mais également pour les comparer de manière objective. Néanmoins, ces mesures ne reflètent pas la réalité, car elles sont généralement basées sur une comparaison d'une image de bonne qualité à une image de qualité artificiellement réduite. Plus de détails sur l'entraînement des algorithmes par apprentissage profond, et des mesures de qualité peuvent se trouver [ici](https://titsitits.github.io/super_resolution/).

### Des licences hétérogènes et parfois contraignantes



### Un domaine en pleine expansion

L'utilisation des réseaux de neurones profonds dans la recherche, et en particulier dans le traitement d'images est un champ de recherche particulièrement en mouvement actuellement. L'état de l'art évolue rapidement. A titre d'exemple, lors de la phase portant sur la comparaison d'algorithmes de super-résolution (voir [ici](https://titsitits.github.io/super_resolution/)), l'algorithme [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (ECCV, 2018)](https://paperswithcode.com/paper/esrgan-enhanced-super-resolution-generative) était premier dans la plupart des classements de super-résolution. Ses performances semblent néanmoins avoir été surpassées depuis, par une autre algorithme: [Second-Order Attention Network for Single Image Super-Resolution (CVPR 2019)](https://paperswithcode.com/paper/second-order-attention-network-for-single).



## Une sélection d'algorithmes libres de droits, utilisables, et compatibles

Dans ce projet, nous avons tenté d'obtenir un compromis entre un pipeline de restauration d'images à la fois générique et efficace. Les techniques ont donc été sélectionnées de manière à fonctionner ensemble dans un même programme, et à donner un résultat  le plus robuste et générique possible.



Pour chacune de ces techniques

, à la comparaison de leurs diverses implémentations par différents algorithmes de Deep Learning, et plus particulièrement à la mise en série 