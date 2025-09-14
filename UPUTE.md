Podaci se nalaze u **data** folderu i u ovom slućaju koristim samo **channel_41.csv** radi jednostavnosti.

U **data_provider/data_loader.py** sam dodala **ESA_ADSegLoader**. Jedina posebnost za razliku od drugih loadera je to što podatke razdvajam pomoću datuma. U ovom slučaju uzela sam vremenski period od 6 mjeseci. obično znam staviti i do godinu dana no postoje slučajevi gdje to bude previše podataka pa sam smanjila za svaki slučaj.

Za pokretanje programa možeš iskoristiti **launch.json** file koji treba biti u **.vscode** folderu preko debbugera ili iskoristi ovu naredbu:
```
python -u run.py \
    --task_name anomaly_detection \ 
    --is_training 1 \
    --model_id TimesNet_ESA-AD_41_y2000 \
    --model TimesNet \
    --data ESA-AD \
    --root_path ./data/ \
    --data_path channel_41.csv 
    --features S \
    --seq_len 1000 \
    --pred_len 0 \
    --d_model 64 \
    --d_ff 8 \
    --e_layers 2 \
    --enc_in 1 \
    --c_out 1 \
    --top_k 3 \
    --batch_size 128 \
    --train_epochs 10 \
    --anomaly_ratio 0.01 \
    --use_gpu True \
    --gpu 0 
```
sa **--gpu** parametrom odaberi koju grafičku koristiš.

Dio parametara uzela sam iz **scripts/anomaly_detection/MSL/TimesNet.sh** i iskreno ima ih toliko da za neke ne znam ni čemu služe ali ako je mreža prevelika ili ima previše podataka slobodno promijeni neki od njih jer ja trenutno ne mogu znati koliko dobro odgovaraju.

Postotak anomalija u skupu je oko 1% što bi po pravilu **--anomaly_ratio** trebao biti također 1, no u Anomaly Transformeru izrečun thresholda je puno bolji kad stavim 0.01 tako da sam to ponovila i ovdje.