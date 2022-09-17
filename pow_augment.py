import random

lang_data_pool = ['انگلش', 'پشتو', 'پنجابی ', 'سندھی'] #add more files related to language
name_data_pool = ['ثاقب', 'عمران', 'زبیر', 'علی', 'قیصر'] #add more files related to names
religion_data_pool = ['اسلام ', 'عیسائی ', 'یہودی ', 'ہندو ','سکھ'] #add more files related to religion

final = [name_data_pool, lang_data_pool, religion_data_pool]

def pow_augs(input_sentence):
    new_sentence = ' '
    aug_sentence = input_sentence.split(' ')
    for ini in aug_sentence:
        for i in range(len(final)):
            if ini in final[i]:
                rez = input_sentence.replace(ini, random.choice(final[i]))
                if rez != input_sentence:
                    print(rez)

for i in range(5):
    pow_augs("قیصر سکھ دے نال پشتو نئیں بول سکدا۔")