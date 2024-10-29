import os
import time
import torch
from gtts import gTTS
from flask import Flask, render_template, request, jsonify
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load the model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Supported languages and their codes
LANGUAGES = {
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Gujarati": "gu_IN",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovene": "sl_SI",
}

app = Flask(__name__)

def translate(text, src_lang, tgt_lang):
    print(f"Translating from {src_lang} to {tgt_lang} with text: '{text}'")
    
    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.lang_code_to_id['en_XX']  # Translate to English first
        )
    
    english_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"Translated to English: '{english_text}'")

    if tgt_lang == 'en_XX':
        return english_text  # Return English directly if it's the target language

    # Now translate from English to target language
    tokenizer.src_lang = 'en_XX'  # Set source to English
    encoded_input = tokenizer(english_text, return_tensors="pt")
    
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded_input,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]  # Translate to target language
        )
    
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(f"Translated text: '{translated_text}'")
    return translated_text

def text_to_speech(text, lang):
    unique_filename = f"output_{int(time.time())}.mp3"  # Unique filename using timestamp
    output_path = os.path.join('static', unique_filename)
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    return unique_filename  # Return just the filename

@app.route('/')
def index():
    return render_template('index.html', LANGUAGES=LANGUAGES)

@app.route('/convert', methods=['POST'])
def convert_text():
    data = request.get_json()
    text = data['text']
    src_lang = data['src_lang']
    tgt_lang = data['tgt_lang']

    print(f"Received text: '{text}', Source Language: '{src_lang}', Target Language: '{tgt_lang}'")

    # Translate the text
    translated_text = translate(text, src_lang, tgt_lang)

    # Convert the translated text to speech
    output_path = text_to_speech(translated_text, tgt_lang.split("_")[0])

    return jsonify({"translated_text": translated_text, "audio_path": output_path})

if __name__ == '__main__':
    app.run(debug=True)
