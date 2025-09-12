# Predictive Maintenance System

Bu proje, makine arızalarını önceden tahmin etmek için tasarlanmış bir web uygulamasıdır.

## Özellikler

- Makine durumu izleme
- Arıza tahmini
- Bakım planlaması
- Veri analizi ve görselleştirme

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT main:app
```

## Deployment

Bu uygulama Render.com üzerinde çalışmaktadır.
