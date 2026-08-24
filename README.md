# Submission 1: credit card fraud detection
Nama: Muhammad Fathurrohman

Username dicoding: M_Fathurrohman

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  terdiri dari 284,807 transaksi selama 2 hari dengan 28 fitur per detik transaksinya. |
| Masalah | Penipuan kartu kredit merupakan masalah serius yang merugikan nasabah dan juga bank. Hal ini merupakan tindakan kriminal yang sangat merugikan bank atau lembaga keuangan |
| Solusi machine learning | maka dari itu dibuat lah model yang dapat mempelajari dan mengenali pola dari transaksi kartu kredit serta memprediksi apakah sebuah transaksi tersebut termasuk penipuan atau tidak. keunggulan modelnya adalah arstitektur modelnya adapatif karena bisa disesuaikan sendiri melalui hyperparamter tuning.|
| Metode pengolahan | Metode pengolahan data yang digunakan adalah Z-score standardization menggunakan `tft.scale_to_z_score()` pada seluruh 30 fitur numerik (V1–V28, Time, Amount) agar setiap fitur memiliki skala yang seragam. Label target `Class` di-cast ke float32 agar kompatibel dengan model. Dataset dibagi dengan rasio 80% training dan 20% evaluasi |
| Arsitektur model | Model yang dibangun adalah Feedforward Deep Neural Network (DNN) berbasis Keras. Terdapat layer input untuk 30 fitur numerik, diikuti 1–3 hidden layer Dense dengan aktivasi ReLU, BatchNormalization, dan Dropout untuk regularisasi. Output layer menggunakan 1 neuron dengan aktivasi Sigmoid untuk klasifikasi biner. Optimizer yang digunakan adalah Adam dengan Binary Crossentropy sebagai loss function. Hyperparameter (jumlah layer, jumlah unit, dropout rate, learning rate) dioptimalkan secara otomatis menggunakan Keras Tuner RandomSearch dengan 3 trial.|
| Metrik evaluasi | Metrik yang digunakan pada model yaitu **BinaryAccuracy**, **Precision**, **Recall**, **TruePositives**, **FalsePositives**, **TrueNegatives**, dan **FalseNegatives** untuk mengevaluasi performa model dalam menentukan klasifikasi.|
| Performa model | Model yang dibuat menghasilkan performa yang cukup baik dalam memberikan prediksi untuk transaksi kartu kredit yang dilakukan dan dari pelatihan yang dilakukan model menghasilkan binary_accuracy di sekitar 99% , terdapat indikasi overfitting karena dari hasil evaluasi dataset hanya sekitar 17% terindikasi penipuan(fraud), dan nilai precisionnya hanya sekitar 82%.  |
