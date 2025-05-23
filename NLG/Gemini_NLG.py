import google.generativeai as genai
from NLG.Translator import Translator
import json

class GeminiNLG:
    def __init__(self, gemini_api_key):
        self.gemini_api_key = gemini_api_key
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash", #gemini-1.5-flash-8b, gemini-1.5-flash, gemini-2.0-flash-exp
            system_instruction=(
                "شما یک ربات چت مالی هستید که فقط به پرسش‌ها و موضوعات مرتبط با امور مالی پاسخ می‌دهید. "
                "اگر کاربر سوالی نامربوط پرسید، مودبانه به او بگویید که فقط به سوالات مالی پاسخ می‌دهید. "
                "همواره تلاش کنید پاسخ‌های شما در کمال ادب و اختصار باشند. "
                "زبان پاسخگویی شما فارسی است."
            )
        )
        self.translator = Translator()


    def detect_intent(self, user_input):
        intent_options = {
            "open_account_free": "افتتاح حساب آزاد",
            "open_account_current": "افتتاح حساب جاری",
            "open_account_deposit": "افتتاح حساب سپرده",
            "loan_free": "وام بدون بهره",
            "loan_interest": "وام با بهره",
            "card2card": "کارت به کارت",
            "paya": "پایا",
            "convert_cheque": "نقد کردن چک",
            "receipt_payment": "پرداخت قبض",
            "installment_payment": "پرداخت اقساط",
            "turnover_bill": "صورت گردش حساب",
            "balance_bill": "صورت مانده",
            "submit_cheque": "ثبت چک",
            "recieve_cheque": "دریافت چک",
            "change_password": "تغییر رمز",
            "duplicate_card": "صدور کارت مجدد",
            "close_card": "ابطال کارت",
            "delegate_account": "حساب حقوقی",
            "currency_request": "درخواست ارز",
            "software_problem": "مشکل نرم افزاری",
            "signin_problem": "مشکل ورود به سامانه",
            "Others": "سایر موارد"
        }

        prompt = f"""
        جمله کاربر را تحلیل کرده و نیتمندی آن را تشخیص دهید:
        "{user_input}"

        از بین گزینه‌های زیر فقط یک مورد که بیشترین ارتباط را با جمله کاربر دارد انتخاب کنید:
        - open_account_free: افتتاح حساب آزاد
        - open_account_current: افتتاح حساب جاری
        - open_account_deposit: افتتاح حساب سپرده
        - loan_free: وام بدون بهره
        - loan_interest: وام با بهره
        - card2card: کارت به کارت
        - paya: پایا
        - convert_cheque: نقد کردن چک
        - receipt_payment: پرداخت قبض
        - installment_payment: پرداخت اقساط
        - turnover_bill: صورت گردش حساب
        - balance_bill: صورت مانده
        - submit_cheque: ثبت چک
        - recieve_cheque: دریافت چک
        - change_password: تغییر رمز
        - duplicate_card: صدور کارت مجدد
        - close_card: ابطال کارت
        - delegate_account: حساب حقوقی
        - currency_request: درخواست ارز
        - software_problem: مشکل نرم افزاری
        - signin_problem: مشکل ورود به سامانه
        - Others: سایر موارد

        پاسخ را دقیقاً به این فرمت برگردانید: 
        exact_keyword, translated_intent
        مثال: 
        loan_free, وام بدون بهره
        """
        response = self.model.generate_content(prompt)
        intent_pair = response.text.strip().split(", ")
        
        if len(intent_pair) == 2 and intent_pair[0] in intent_options:
            return intent_pair[0], intent_pair[1]
        else:
            return "Others", "سایر موارد"



    def generate_slot_response(self, previous_question, user_input, intent, slot, ir_examples, guided_mode=False, guided_mode_examples=None):
        self.intent, fa_slot = self.translator.translate(intent, slot)

        if guided_mode == False:
            # Second step: Generate slot question
            question_prompt = f"""
            شما در جایگاه دستیار بانکی هستید.
            موضوع گفتگو: {self.intent}
            اسلات مورد نیاز: {fa_slot}
            
            با توجه به مثال‌های زیر (که کاربر نمی‌بیند)، سوالی طراحی کن که:
            1. مستقیماً درباره دریافت مقدار {fa_slot} باشد
            2. کاربر را طوری راهنمایی کند که پاسخش شبیه مثال‌ها باشد
            3. از خود مثال‌ها مستقیماً استفاده نشود، اما فرمت پاسخ را توضیح دهد
            
            مثالها:
            {ir_examples}
            
            ساختار پاسخ نهایی:
            1. سوال طراحی شده فقط برای دریافت {fa_slot} و نه چیزی دیگری. تمرکز سوال باید فقط روی این اسلات باشد.
            2. یک راهنمای کوتاه با الهام از مثالها (بدون کپی کردن از مثالها) فقط برای اسلات مربوطه {fa_slot}
            پاسخ را به فارسی و با فرمت طبیعی سوال-راهنما ارائه دهید.

            توجه کن که کاربر نهایی فقط پیام تو را می بیند و موارد غیر ضروری را حذف کن.
            """
            question_response = self.model.generate_content(question_prompt)
            
            # Combine responses
            final_response = f"از پاسخ شما متشکرم. \n\n{question_response.text}"
            return final_response, fa_slot
        
        else:
            # First step: Check response relevance
            relevance_prompt = f"""
            شما در جایگاه دستیار بانکی هستید.
            موضوع گفتگو: {self.intent}
            
            سوال قبلی: {previous_question}
            پاسخ کاربر: "{user_input}"
            
            آیا این پاسخ مرتبط با سوال است؟ 
            - اگر مرتبط است، فقط و فقط یک تشکر مختصر بیان کن (بدون بیان مرتبط بودن یا علت آن)
            - اگر مرتبط نیست، در 2-1 جمله دلیل عدم ارتباط را توضیح بده
            پاسخ را به فارسی بنویس و از سایر متن‌ها استفاده نکن.
            """
            relevance_response = self.model.generate_content(relevance_prompt)


            # generate guided question ..
            question_prompt = f"""
            شما در جایگاه دستیار بانکی هستین.
            موضوع گفتگو: {self.intent}
            اسلات مورد نیاز: {fa_slot}

            کاربر در پاسخ به سوال {previous_question} به شکل مناسبی پاسخ نداده است.
            با توجه به مثال‌های زیر 
            {ir_examples}

            و اسلات های مختلف در این مثال ها 
            {guided_mode_examples}

            1-
            سوالی طرح کنید که مقدار {fa_slot} را از کاربر به شکل مناسب بپرسد.
            تمرکز فقط روی اسلات {fa_slot} است.
            هیچ چیز اضافی نباید از کاربر پرسیده شود.

            2-
            و کاربر را راهنمایی کنید که چگونه باید برای {fa_slot} پاسخ بدهد.
            از روی مثال ها کپی برداری نشود.
            اما بخشی از مثال ها می توانند برای راهنمایی کاربر به عنوان پاسخ مربوطه برای {fa_slot} استفاده شوند.
            در نهایت کاربر باید طوری راهنمایی شود که اسلات مدنظر{fa_slot} را مانند مثال ها بیان کند.

            توجه کن که کاربر نهایی فقط پیام تو را می بیند و موارد غیر ضروری را حذف کن.
            """
            question_response = self.model.generate_content(question_prompt)
            
            # Combine responses
            final_response = f"guided mode: {relevance_response.text}\n\n{question_response.text}"
            return final_response, fa_slot
            

    def update_dict_based_on_input(self, input_dict, user_input):
        prompt = f"""
        قالب و مقدار زیر در نظر بگیر:
        {input_dict}

        با توجه به گفته کاربر
        {user_input}
        فیلد مربوطه را پیدا کرده و سپس 
        عینا مشابه و کپی شده 
        {input_dict}
        را فقط به صورت بروزرسانی شده با قالب یکسان خروجی بده.

        دقت کن که کلیدها و قالب باید دقیقا یکسان باشد و فقط مقدار مدنظر تغییر کرده باشد.
        هیچ توضیحی در کنار خروجی لازم نیست.
        """
        # print(f'prompt: {prompt}')
        try:
            response = self.model.generate_content(prompt)
            print(f'update_dict_based_on_input response: {response.text}')

            sanitized_response = response.text.strip()
            if sanitized_response.startswith("```json") and sanitized_response.endswith("```"):
                sanitized_response = sanitized_response[7:-3].strip()

            sanitized_response = sanitized_response.replace("'", '"')

            # Json pre-process 
            if sanitized_response.startswith("{") and sanitized_response.endswith("}"):
                updated_dict = json.loads(sanitized_response)
            else:
                print(f"Invalid JSON format in response: {sanitized_response}")
                return None

            if isinstance(updated_dict, dict) and all(key in updated_dict for key in input_dict.keys()):
                return updated_dict
            else:
                print(f"Updated dictionary keys do not match the original. Returning None.")
                return None

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error processing the model response: {e}")
            return None
