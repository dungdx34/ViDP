Cài đặt các thư viện sau:
python 3.6,
pytorch 1.5.1,
gensim,
requests,
tqdm,
flask

Hướng dẫn chạy:
Tải thư mục models (https://drive.google.com/file/d/1K3nqOhaEt9xFDPcC-rJGPLokjL3vHL16/view?usp=sharing)  và giải nén vào project.
Tải thư viện VncoreNLP về máy tại địa chỉ: https://github.com/vncorenlp/VnCoreNLP
Chỉnh các đường dẫn trong file api/configs.py

Chạy các lệnh:
cd api
python dp.py

Khi đó api dependency parser đã được chạy với cổng như cấu hình trong file config (mặc định là 8880).
Chạy test api bằng câu lệnh: python test.py





