persian_translations = {
    # Intent translations
    "intents": {
        "open_account_free": "افتتاح حساب آزاد",
        "open_account_current": "افتتاح حساب جاری",
        "open_account_deposit": "افتتاح حساب سپرده",
        "loan_free": "درخواست وام بدون بهره",
        "loan_interest": "درخواست وام با بهره",
        "card2card": "انتقال کارت به کارت",
        "paya": "پایا",
        "convert_cheque": "نقد کردن چک",
        "receipt_payment": "پرداخت قبض",
        "installment_payment": "پرداخت اقساط",
        "turnover_bill": "صورت حساب گردش",
        "balance_bill": "صورت حساب مانده",
        "submit_cheque": "ثبت چک",
        "recieve_cheque": "دریافت چک",
        "change_password": "تغییر رمز",
        "duplicate_card": "صدور مجدد کارت",
        "close_card": "مسدودسازی کارت",
        "delegate_account": "تفویض حساب",
        "currency_request": "درخواست ارز",
        "software_problem": "مشکل نرم افزاری",
        "signin_problem": "مشکل ورود به سیستم"
    },
    
    # Slot translations
    "slots": {
        # Common slots
        "fname": "نام",
        "lname": "نام خانوادگی",
        "national_id": "کد ملی",
        "father_name": "نام پدر",
        "birth_date": "تاریخ تولد",
        "address": "آدرس",
        "card_number": "شماره کارت",
        "current_pass": "رمز فعلی",
        "new_pass": "رمز جدید",
        
        # Account related
        "issuance_card": "صدور کارت",
        "activate_ib": "فعالسازی بانکداری اینترنتی",
        "starter_amount": "میزان سپرده اولیه",
        "support": "مدارک ضمیمه",
        "cheque_n": "شماره چک",
        "shared_cheque": "حساب چک مشترک",
        "benefit_rate": "نرخ سود",
        "deposit_duration": "مدت سپرده",
        
        # Loan related
        "loan_reason": "دلیل دریافت وام",
        "zfname": "نام ضامن",
        "zlname": "نام خانوادگی ضامن",
        "znational_id": "کد ملی ضامن",
        "insurance_req": "نیاز به بیمه‌نامه",
        "loan_amount": "مبلغ وام",
        "loan_duration": "مدت وام",
        "loan_benefit_rate": "نرخ بهره وام",
        "loan_support": "وثیقه وام",
        
        # Transfer related
        "transfer_amount": "مبلغ انتقال",
        "transfer_datetime": "زمان انتقال",
        "transfer_reason": "دلیل انتقال",
        "cvv2": "CVV2",
        "trans_pass": "رمز تراکنش",
        "receiver_card": "کارت مقصد",
        "static_pass": "رمز ثابت",
        "trans_periodic": "انتقال دوره ای",
        "receiver_iban": "شبا مقصد",
        "receiver_bank": "بانک مقصد",
        
        # Cheque related
        "cfname": "نام صادرکننده چک",
        "clname": "نام خانوادگی صادرکننده چک",
        "cnational_id": "کد ملی صادرکننده چک",
        "sayad_id": "شناسه صیاد",
        "cheque_date": "تاریخ چک",
        "cheque_amount": "مبلغ چک",
        "cheque_reason": "دلیل چک",
        
        # Payment related
        "bill_id": "شناسه قبض",
        "payment_id": "شناسه پرداخت",
        "phone_number": "شماره تلفن",
        "post_code": "کد پستی",
        "installment_amount": "مبلغ قسط",
        "loan_id": "شناسه وام",
        "installment_n": "شماره قسط",
        
        # Account management
        "account_id": "شناسه حساب",
        "start_datetime": "زمان شروع",
        "end_datetime": "زمان پایان",
        "trans_n": "شماره تراکنش",
        "min_amount": "حداقل مبلغ",
        "max_amount": "حداکثر مبلغ",
        "balance_datetime": "زمان بررسی مانده",
        "name": "نام نماینده",
        "b-ncid": "کد ملی ذینفع",
        "advocacy_reason": "دلیل تفویض",
        "renew_reason": "دلیل تمدید",
        "blocking_reason": "دلیل مسدودسازی",
        
        # Currency
        "country": "کشور",
        "amount": "مقدار",
        "currency": "ارز"
    }
}

class Translator:
    def __init__(self):
        self.static_translations = persian_translations

    def translate(self, intent=None, slot=None):
        print(f"intent:{intent}, slot:{slot}")
        intent_fa = self.static_translations['intents'].get(intent, "")
        slot_fa = self.static_translations['slots'].get(slot, "")
        print(f'translated: {intent_fa}, {slot_fa}')
        # if not intent_fa or not slot_fa:
        #     return self._model_translate(intent, slot)
        return intent_fa, slot_fa

    def translate_dict_to_persian(self, data):
        translated_intent = self.static_translations['intents'].get(data['intent'], data['intent'])
        translated_params = {}
        for key, value in data['parameters'].items():
            translated_key = self.static_translations['slots'].get(key, key)
            translated_params[translated_key] = value
        return {
            'intent': translated_intent,
            'parameters': translated_params
        }

    def translate_dict_to_english(self, data):
        english_intent = data['intent']
        for eng_key, per_val in self.static_translations['intents'].items():
            if per_val == english_intent:
                english_intent = eng_key
                break
        
        english_params = {}
        for per_key, value in data['parameters'].items():
            eng_slot = per_key
            for eng_slot_key, per_slot_val in self.static_translations['slots'].items():
                if per_slot_val == per_key:
                    eng_slot = eng_slot_key
                    break
            english_params[eng_slot] = value
        
        return {
            'intent': english_intent,
            'parameters': english_params
        }