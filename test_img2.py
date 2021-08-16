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
from yargy.predicates import caseless, normalized, dictionary, gte, lte, true, type, in_, eq, length_eq
from yargy.interpretation import fact
from yargy.tokenizer import MorphTokenizer
from natasha import (
    MoneyExtractor,
    OrganisationExtractor
)
# импорт вспомогательных файлов
from lib.imageoptimize import process_image_for_ocr
from lib.deskew import deskew

# собранные данные
result = {}
# количество обработанных документов
docs_index = 0

# директория с изображениями счетов
DOCS_dir = "test"

# получаем список файлов
docs = os.listdir(DOCS_dir)

# файл в который будем складывать распознанный текст
outfile = "out_text.txt"
outfile_json = "out.json"

# открываем файл для записи текста
f = open(outfile, "a")

# правила для yargy
INT = type('INT')
EOL = type('EOL')
# правило по длине для ИНН
INN = rule(
    and_(
        INT,
        or_(
            length_eq(10),
            length_eq(12)
        )
    )
)
# правило по длине для КПП и БИК
KPP_BIK = rule(
    and_(
        INT,
        length_eq(9)
    )
)
# правило по длине для счетов
INVOICE = rule(
    and_(
        INT,
        length_eq(20)
    )
)
# правило для ячейки БИК и Счета
BIK_INVOICE = rule(
    KPP_BIK,
    EOL,
    INVOICE
)
# для парсинга сумм из строк
MONEY = or_(
    rule(
    INT.repeatable(),
    in_(',.-'),
    INT
    ),
    rule(INT)
)

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

# парсеры для ИНН, БИК, КПП, счетов
parser_inn = Parser(INN)
parser_kpp_bik = Parser(KPP_BIK)
parser_invoce = Parser(INVOICE)
parser_bik_invoice = Parser(BIK_INVOICE)

# парсер для сумм
parser_money = Parser(MONEY)
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
org_extractor = OrganisationExtractor()

# проходим по списку файлов
for doc in docs:

    # временная переменная для сбора данных
    tmp = {}

    # обработка изображения и парсинг таблиц
    img = DOCS_dir + "\\" + doc
    im1 = cv2.imread(img, 0)
    im = cv2.imread(img)

    optimized_img = process_image_for_ocr(img)
    cv2.imwrite('optimized.jpg', optimized_img)

    ret, thresh_value = cv2.threshold(im1, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,5),np.uint8)
    dilated_value = cv2.dilate(thresh_value, kernel, iterations = 1)

    # contours, hierarchy = cv2.findContours(dilated_value,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cordinates = []
    # количество найденных, подходящих контуров
    count = 0
    # индекс контура
    index = 0
    # сборка данных по контрагенту
    counterparty = {}
    bank = {}
    rsch = {}
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
            table_orig = im[y:y+h, x:x+w]
            table = dilated_value[y:y+h, x:x+w]
            # спарсим текст с подходящих областей
            text = str(((pytesseract.image_to_string(table_orig, lang="rus"))))
            cv2.imwrite('tmp_img\\table'+str(count)+'.jpg', table_orig)
            count += 1
            # print("\n----------------------------\n")
            # print(text)
            # print("\n----------------------------\n")
            # плохо! если в тексте есть БИК, то это нужный нам кусок
            if text.find('БИК') != -1 or text.find('Получатель') != -1 or text.find('Банк') != -1:
                # print(text)
                # переносим область в новый рисунок, просто для инфы
                cv2.imwrite('tmp_img\\table'+str(count)+'.jpg', table)
                count += 1
                # находим контуры в таблице
                contours_t, hierarchy_t = cv2.findContours(table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cordinates_table = []
                count_table = 0
                c_t = 0
                is_first = True
                for cnt_t in contours_t:
                    # пропускаем первую итерацию, так как там оригинал таблицы
                    if is_first:
                        is_first = False
                        continue
                    x,y,w,h = cv2.boundingRect(cnt_t)
                    cordinates_table.append((x,y,w,h))
                    # bounding the images
                    if cv2.contourArea(cnt_t) > 10000 :
                        # берем ячейку таблицы
                        cell = table_orig[y:y+h, x:x+w]
                        cv2.imwrite('tmp_img\\cell'+str(c_t)+'.jpg', cell)
                        c_t += 1
                        # смотрим на текст ячейки
                        text = str(((pytesseract.image_to_string(cell, lang="rus"))))
                        print([_.value for _ in tokenizer(text)])
                        if text.lower().find('получатель') > -1 and text.lower().find('банк') < 0 :
                            # print(text)
                            # print(text.split("\n"))
                            for line in text.split("\n"):
                                # print("\n----------------------------\n")
                                # print(line)
                                # print("\n----------------------------\n")
                                if line.lower().find('получатель') < 0 and line != '' :
                                    counterparty["Наименование"] = line
                                    # tmp["Наименование"] = line
                            # print([_.value for _ in tokenizer(text)])
                            #matches = org_extractor(text)
                            #for match in matches:
                            #    print(match.span, match.fact)
                        if text.lower().find('банк') > -1 and text.lower().find('получателя') > -1 :
                            for line in text.split("\n"):
                                # print("\n----------------------------\n")
                                # print(line)
                                # print("\n----------------------------\n")
                                if line.lower().find('банк') > -1 and line.lower().find('банк получателя') < 0 and line != '' :
                                    bank["Наименование"] = line
                            # print(text)
                            # print(text.split("\n"))
                            # print([_.value for _ in tokenizer(text)])
                            #matches = org_extractor(text)
                            #for match in matches:
                            #    print(match.span, match.fact)
                        # если в строке есть переносы строк, то это ячейка с БИК и счетом
                        if text.find('\n') > -1 :
                            for match in parser_bik_invoice.findall(text):
                                bank["БИК"] = match.tokens[0].value
                                # tmp['bik'] = match.tokens[0].value
                                bank["КорреспондентскийСчет"] = match.tokens[2].value
                                # tmp['invoice'] = match.tokens[2].value
                                # смотрим, что нашлось
                                # print(match.span, [_.value for _ in match.tokens])
                        else :
                            for match in parser_invoce.findall(text):
                                rsch["НомерСчета"] = match.tokens[0].value
                                # tmp['recipient_invoice'] = match.tokens[0].value
                                # смотрим, что нашлось
                                # print(match.span, [_.value for _ in match.tokens])
                            for match in parser_inn.findall(text):
                                    counterparty["ИНН"] = match.tokens[0].value
                                    # tmp['inn'] = match.tokens[0].value
                                    # смотрим, что нашлось
                                    # print(match.span, [_.value for _ in match.tokens])
                            for match in parser_kpp_bik.findall(text):
                                counterparty["КПП"] = match.tokens[0].value
                                # tmp['kpp'] = match.tokens[0].value
                                # смотрим, что нашлось
                                # print(match.span, [_.value for _ in match.tokens])
                        if text.lower().find('инн') > -1 :
                            for match in parser_inn.findall(text):
                                    counterparty["ИНН"] = match.tokens[0].value
                                    # tmp['inn'] = match.tokens[0].value
                                    # смотрим, что нашлось
                                    # print(match.span, [_.value for _ in match.tokens])
                        if text.lower().find('кпп') > -1 :
                            for match in parser_kpp_bik.findall(text):
                                counterparty["КПП"] = match.tokens[0].value
                                # tmp['kpp'] = match.tokens[0].value
                                # смотрим, что нашлось
                                # print(match.span, [_.value for _ in match.tokens])
                        
                        # print(text)
                        # print("\n------------------------------------\n")
                        # рисуем границы для информации
                        # cv2.rectangle(table_orig, (x,y), (x+w,y+h), (0,0,255), 2)
                # для инфы смотрим как разбилась по контурам таблица
                cv2.imwrite('tmp_img\\detecttable_table.jpg', table_orig)

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
    cv2.imwrite('tmp_img\\detecttable.jpg',im)

    rsch["Банк"] = bank
    counterparty["РасчетныйСчет"] = rsch
    tmp["Контрагент"] = counterparty

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
        # print(match.span, [_.value for _ in match.tokens])
        # получаем подстроку из текста
        # print(text[slice(*match.span)])
        tmp['НазначениеПлатежа'] = text[slice(*match.span)].rstrip()

    # парсим строку "ИТОГО"
    total = 0
    for match in parser_total.findall(text):
        # смотрим, что нашлось
        # print(match.span, [_.value for _ in match.tokens])
        # получаем подстроку из текста и удаляем переносы строк
        # print(text[slice(*match.span)].rstrip())
        cut_text = text[slice(*match.span)].rstrip()
        for match in parser_money.findall(cut_text):
            # print(match.span, [_.value for _ in match.tokens])
            total = cut_text[slice(*match.span)].rstrip()
            total = total.replace("-", ".").replace(",", ".").replace(" ", "")
            tmp['СуммаДокумента'] = float(total)

    # парсим строку "НДС"
    vat = False
    for match in parser_vat.findall(text):
        # смотрим, что нашлось
        # print(match.span, [_.value for _ in match.tokens])
        # получаем подстроку из текста и удаляем переносы строк
        # print(text[slice(*match.span)].rstrip())
        cut_text = text[slice(*match.span)].rstrip()
        for match in parser_money.findall(cut_text):
            # print(match.span, [_.value for _ in match.tokens])
            vat = cut_text[slice(*match.span)].rstrip()
            vat = vat.replace("-", ".").replace(",", ".").replace(" ", "")
            vat = float(vat)
    if vat : 
        tmp["РасшифровкаПлатежа"] = [{
            "СуммаПлатежа": total,
            "СтавкаНДС": "НДС20",
            "СуммаНДС": vat
        }]
    else :
        tmp["РасшифровкаПлатежа"] = [{
            "СуммаПлатежа": total,
            "СтавкаНДС": "БезНДС",
            "СуммаНДС": 0
        }]

    # записываем полученное в файл
    # f.write(text)

    result[docs_index] = tmp
    docs_index += 1

print(result)
# закрываем файл
# f.close()