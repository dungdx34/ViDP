Cài đặt các thư viện sau:

python 3.6 <br />
pytorch 1.4.0 <br />
transformers 3.4.0 <br />
gensim <br />
requests <br />
tqdm <br />
overrides <br />
flask

Hướng dẫn chạy:

Tải thư mục models (https://drive.google.com/file/d/1K3nqOhaEt9xFDPcC-rJGPLokjL3vHL16/view?usp=sharing)  và giải nén vào project. <br />
Tải thư viện VncoreNLP về máy tại địa chỉ: https://github.com/vncorenlp/VnCoreNLP <br />
Chỉnh các đường dẫn trong file api/configs.py <br />

Chạy các lệnh: <br />
cd api <br />
python dp.py <br />

Khi đó api dependency parser đã được chạy với cổng như cấu hình trong file config (mặc định là 8880). <br />
Chạy test api bằng câu lệnh: python test.py <br />





