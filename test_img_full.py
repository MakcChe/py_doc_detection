# импорт библиотек
from PIL import Image
import pytesseract
import sys
from pdf2image import convert_from_path
import os
# для поиска таблиц
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
# для поиска по тексту
from yargy import Parser, rule, and_, or_, not_
from yargy.predicates import caseless, normalized, dictionary, gte, lte, true, type, in_, eq
from yargy.interpretation import fact
from yargy.tokenizer import MorphTokenizer
from natasha import (
    MoneyExtractor,
)

img = 'test\\test.png'
im1 = cv2.imread(img, 0)
im = cv2.imread(img)

ret,thresh_value = cv2.threshold(im1, 180, 255, cv2.THRESH_BINARY_INV)
# вырезает часть изображения
# roi = im1[100:1000, 100:1000]
# cv2.imwrite('roi.jpg',roi)

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3) 

rho = 1  # разрешение по расстоянию в пикселях сетки Хафа
theta = np.pi / 180  # угловое разрешение в радианах сетки Хафа
threshold = 15  # минимальное количество голосов (пересечений в ячейке сетки Хафа)
min_line_length = 100  # минимальное количество пикселей, составляющих линию
max_line_gap = 20  # максимальный зазор в пикселях между соединяемыми отрезками
line_image = np.copy(im) * 0  # создание бланка для рисования линий
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
lines_edges = cv2.addWeighted(im, 0.8, line_image, 1, 0)
# cv2.imwrite('123.jpg',lines_edges)
# lines = cv2.HoughLines(edges,1,np.pi/180, 200)

kernel = np.ones((5,5),np.uint8)
dilated_value = cv2.dilate(thresh_value, kernel, iterations = 1)

# contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cordinates = []
# количество найденных, подходящих контуров
count = 0
# индекс контура
index = 0
# создаем маску изображения
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cordinates.append((x,y,w,h))
    # если длина или ширина больше 2000 пикселей
    # if w > 2000 or h > 2000:
    # если площадь выделенной по контуру области больше 30000 пикселей, для таблиц
    if cv2.contourArea(cnt) > 30000 :
        arc_len = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.1 * arc_len, True)
        # указываем область, которую хотим взять из изображения
        roi = im[y:y+h, x:x+w]
        # спарсим текст с подходящих областей
        text = str(((pytesseract.image_to_string(roi, lang="rus"))))
        # плохо! если в тексте есть БИК, то это нужный нам кусок
        if text.find('БИК') != -1:
            # переносим область в новый рисунок, просто для инфы
            cv2.imwrite('table'+str(count)+'.jpg', roi)
            count += 1

        # пробую вырезать не прямоугольником, а точно по контуру
        # добавляем маску к исходному изображению
        # mask = np.zeros_like(im1)
        # рисуем контур на маске
        # cv2.drawContours(mask, contours, index, 255, -1)
        # извлекаем объект и помещаем в выходное изображение
        # out = im1*mask
        # cv2.imwrite('table'+str(count)+'.jpg',out)

        # просто для информации пробую вывести оригинальные контуры и прямоугольный контур
        # рисуем контур
        cv2.drawContours(im, contours, index, ( 255, 0, 0 ), 2)
        # рисуем прямоугольники вокруг по контурам
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,0,255), 2)
    # индекс текущего контура
    index += 1
# plt.imshow(im)
# cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
cv2.imwrite('detecttable.jpg',im)

# пробую разобрать вытащенные по контуру детали, аналогично с предыдущим вариантом, только теперь площади контуров не проверяем
table_img = 'test_table.jpg'
im = cv2.imread(table_img, 0)
ret,thresh_value = cv2.threshold(im, 180, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5,5),np.uint8)
dilated_value = cv2.dilate(thresh_value, kernel, iterations = 1)
contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cordinates = []
count = 0
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cordinates.append((x,y,w,h))
    #bounding the images
    if w > 100 or h > 100:
        # roi = im[y:y+h, x:x+w]
        # cv2.imwrite('table'+str(count)+'.jpg',roi)
        # count += 1
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),1)
cv2.imwrite('detecttable_table.jpg',im)


# директория с изображениями счетов
DOCS_dir = "test"

# получаем список файлов
docs = os.listdir(DOCS_dir)

# файл в который будем складывать распознанный текст
outfile = "out_text.txt"

# открываем файл для записи текста
f = open(outfile, "a")

# правила для yargy
INT = type('INT')
# для парсинга сумм из строк
MONEY = rule(
    INT.repeatable(),
    in_(',.-'),
    INT
)
parser_money = Parser(MONEY)
# год
YEAR = and_(
    gte(1),
    lte(3000)
)
# день
DAY = and_(
    gte(1),
    lte(31)
)
MONTHS = {
    'январь',
    'февраль',
    'март',
    'апрель',
    'мая',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь'
}
MONTH_NAME = dictionary(MONTHS)
# строка "Счет на оплату ..." может заканчиваться на дату без "г.", либо с "г."
YEAR_WORDS = or_(
    rule(YEAR, caseless('г'), '.'),
    rule(YEAR, normalized('год')),
    rule(YEAR)
)
# дата
DATE = rule(
    DAY,
    MONTH_NAME,
    # YEAR
    YEAR_WORDS
)
# правило для парсера подстроки "Счет на оплату ..."
INVOCE_TITLE = rule(
    caseless('счет'),
    true().repeatable(max=15),
    DATE
)

# правило для парсера подстроки "Итого"
TOTAL = rule(
    or_(
        rule(caseless('итого'), ':'),
        rule(caseless('итого'), caseless('к'), caseless('оплате'), ':')
    ),
    MONEY
    # любые символы, не больше 5 шт.
    # true().repeatable(max=5), 
    # любые символы, кроме конца строки, любое количество
    # not_(type('EOL')).repeatable(),
    # конец строки
    # type('EOL')
)

# правило для парсера подстроки "НДС"
VAT = rule(
    or_(
        rule(caseless('НДС'), eq(':').optional()), 
        rule('(', caseless('НДС'), ')', eq(':').optional())
    ),
    MONEY
    # любые символы, не больше 5 шт.
    # true().repeatable(max=5), 
    # любые символы, кроме конца строки, любое количество
    # not_(type('EOL')).repeatable(),
    # конец строки
    # type('EOL')
)

# парсер для "Счет на оплату.."
parser_invoce_title = Parser(INVOCE_TITLE)
# парсер для "Итого"
parser_total = Parser(TOTAL)
# парсер для "НДС"
parser_vat = Parser(VAT)

# просто для проверки как токенизируется текст счета
tokenizer = MorphTokenizer()

# готовый парсер из библиотеки natasha
extractor  = MoneyExtractor()

# проходим по списку файлов
for doc in docs:

    # с помощью тессеракта распознаем текст
    text = str(((pytesseract.image_to_string(Image.open(DOCS_dir + "\\" + doc), lang="rus"))))

    # убираем переносы строк с дефисами из полученного текста, на всякий случай
    text = text.replace('-\n', '')

    # for line in text.splitlines():
        # print([_.value for _ in tokenizer(line)])
        # print(list(tokenizer(line)))

    # парсим подстроку "Счет ..."
    for match in parser_invoce_title.findall(text):
        # смотрим, что нашлось
        print(match.span, [_.value for _ in match.tokens])
        # получаем подстроку из текста
        print(text[slice(*match.span)])

    # парсим строку "ИТОГО"
    for match in parser_total.findall(text):
        # смотрим, что нашлось
        print(match.span, [_.value for _ in match.tokens])
        # получаем подстроку из текста и удаляем переносы строк
        # start, stop = match.span
        # print(text[start:stop])
        print(text[slice(*match.span)].rstrip())
        cut_text = text[slice(*match.span)].rstrip()
        for match in parser_money.findall(cut_text):
            print(match.span, [_.value for _ in match.tokens])

    # парсим строку "НДС"
    for match in parser_vat.findall(text):
        # смотрим, что нашлось
        print(match.span, [_.value for _ in match.tokens])
        # получаем подстроку из текста и удаляем переносы строк
        print(text[slice(*match.span)].rstrip())

    # записываем полученное в файл
    # f.write(text)

# закрываем файл
f.close()
