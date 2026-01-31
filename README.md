# ysaBut
Göz çevremizi algılamamızı,okuyabilmemizi ve etrafta güvenli bir şekilde dolaşabilmemizi sağlayan önemli bir duyu organıdır.Bu özelliklerini kaybetmesine neden olan pek çok hastalıktan korunmaya dikkat edilmesi gerekir.Uzun süreler dijital ekranlara bakmak,lazerlerleri dikkatsizce kullanmak ve sivri cisimlerin yakınında dikkatsiz olmak gözlerimize büyük zararlar verebilir.Bu verisetinde Retinitis Pigmentosa,Retina Dekolmanı,Pterjium,Miyopi,Maküler Skar,Glokom, Disk Ödemesi,Diyabetik Retinopati,Santral Seröz Korioretinopati ve Sağlıklı göz görüntüsü olmak üzere 10 sınıf bulunmaktadır.Sınıflar arası fotoğraf sayısı eşit olmadığından dolayı ağırlaştırılmış ağırlıklar kullandım doğru tespit sayısını azaltıyordu.

Bu model Pytorch kullanılarak oluşturulmuş 5 katmanlı bir resim sınıflandırma modelidir.

Bu modelde Kullanılan Yöntemler:
Verisetinden alınan fotoğraflar 256x256 boyutuna getirilip eğitim ve test için %80-%20 bölünmüştür.Eğitim resimlerine rastgele yatay döndürme uygulanmıştır.
Tekrar edilebilirlik için sabit seed kullanılmıştır (42) .
Normalization Stat'ı için ImageNet modelinin statları kullanılmıştır.
Loss: CrossEntropyLoss .
Optimizer : AdamW .
Cuda uyumlu cihazların varlığına göre gpu/cpu üstünde çalışır. (Amd Marka Ekran kartı üstünde Arch linux dağıtımıyla ROCm kullanıldı).
Gradyanların birikim hatalarına sebep olmaması için optimizer.zero_grad() kullanıldı.
Data Loader'a 4 işlemci çekirdeği atanmıştır.

Tkinter'le oluşturulan arayüzü kullanarak istediğiniz göz taraması fotoğrafını kulanarak modelin sınıflandırma yapmasını sağlayabilirsiniz.

Sonuçlar:
<img width="560" height="454" alt="image" src="https://github.com/user-attachments/assets/728c6608-aed6-4899-ad94-9daaf7e1848c" />
<img width="560" height="454" alt="image" src="https://github.com/user-attachments/assets/8ec4e46c-07bc-4116-aa30-02ad37a514ad" />
<img width="560" height="454" alt="image" src="https://github.com/user-attachments/assets/9bea41d9-66c4-44af-90ee-331ec2b254dc" />
Değerlendirme Metrikleri: Doğruluk,Kayıp ve F1

Bağımlılıklar:
Python 3.9+
PyTorch (ROCm uyumlu sürüm kullanılmıştır)
Torchvision
TorchMetrics
Matplotlib
Pillow (PIL)
Tkinter
