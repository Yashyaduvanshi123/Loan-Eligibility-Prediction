import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from colorama import Fore, Style, init
import os

init(autoreset=True)

print("="*70)
print(Fore.CYAN + "Smart Loan Eligibility Prediction System (Hybrid Model)")
print(Style.RESET_ALL + "Developed by:" + Fore.YELLOW + " Aditya Sharma, Ritu Raj Singh Parihar, Yash Yaduvanshi & Pratham Soni")
print(Style.RESET_ALL + "Course:" + Fore.MAGENTA + " Data Science (CSE0521) | PBL Project")
print("="*70 + "\n")

data = {
    'Credit_History': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
                       1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    'Income': [60000, 30000, 80000, 40000, 20000, 75000, 90000, 25000, 65000, 35000,
               120000, 150000, 250000, 500000, 100000, 1000000, 700000, 450000, 30000, 1500000],
    'Employment_Type': ['Salaried', 'Self-Employed', 'Salaried', 'Salaried', 'Self-Employed',
                        'Salaried', 'Salaried', 'Self-Employed', 'Salaried', 'Self-Employed',
                        'Salaried', 'Salaried', 'Self-Employed', 'Salaried', 'Self-Employed',
                        'Salaried', 'Salaried', 'Salaried', 'Self-Employed', 'Salaried'],
    'Loan_Amount': [150000, 100000, 250000, 120000, 80000, 200000, 180000, 90000, 160000, 110000,
                    100000, 80000, 200000, 150000, 100000, 50000, 80000, 100000, 90000, 10000],
    'Dependents': [2, 3, 1, 2, 4, 2, 1, 3, 2, 3,
                   2, 1, 3, 2, 2, 1, 2, 1, 4, 2],
    'Loan_Status': ['Eligible', 'Not Eligible', 'Eligible', 'Eligible', 'Not Eligible',
                    'Eligible', 'Eligible', 'Not Eligible', 'Eligible', 'Not Eligible',
                    'Eligible', 'Eligible', 'Not Eligible', 'Eligible', 'Not Eligible',
                    'Eligible', 'Eligible', 'Eligible', 'Not Eligible', 'Eligible']
}

df = pd.DataFrame(data)


df['Employment_Type'] = df['Employment_Type'].map({'Salaried': 1, 'Self-Employed': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Eligible': 1, 'Not Eligible': 0})

df['Loan_to_Income_Ratio'] = df['Loan_Amount'] / df['Income']


X = df[['Credit_History', 'Income', 'Employment_Type', 'Loan_Amount', 'Dependents', 'Loan_to_Income_Ratio']]
y = df['Loan_Status']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)


print(Fore.CYAN + "\n=== Loan Eligibility Check (Based on ML + Real Criteria) ===\n" + Style.RESET_ALL)

try:
    credit_history = int(input("Enter Credit History (1 = Good, 0 = Bad): "))
    income = int(input("Enter Annual Income (in ₹): "))
    employment_type = input("Enter Employment Type (Salaried / Self-Employed): ").strip()
    loan_amount = int(input("Enter Loan Amount (in ₹): "))
    dependents = int(input("Enter Number of Dependents: "))

 
    if employment_type.lower() not in ['salaried', 'self-employed']:
        print(Fore.RED + "Invalid employment type! Please enter 'Salaried' or 'Self-Employed'.")
        exit()

    employment_type = 1 if employment_type.lower() == "salaried" else 0


    loan_to_income_ratio = round(loan_amount / income, 2)

    user_data = pd.DataFrame({
        'Credit_History': [credit_history],
        'Income': [income],
        'Employment_Type': [employment_type],
        'Loan_Amount': [loan_amount],
        'Dependents': [dependents],
        'Loan_to_Income_Ratio': [loan_to_income_ratio]
    })

   
    user_prediction = model.predict(user_data)
    ml_result = "Eligible" if user_prediction[0] == 1 else "Not Eligible"

  
    reason = ""
    eligible = True

  
    if credit_history == 0:
        eligible = False
        reason += "Bad credit history. "


    if loan_to_income_ratio > 2.5:
        eligible = False
        reason += "Loan amount too high compared to income. "


    if dependents > 3:
        eligible = False
        reason += "Too many dependents affecting repayment capacity. "


    if income < 250000:
        eligible = False
        reason += " Annual income too low for home loan. "


    if eligible and ml_result == "Eligible":
        final_result = "Eligible"
        color = Fore.GREEN
        reason = " Meets all eligibility criteria."
    else:
        final_result = "Not Eligible"
        color = Fore.RED
        if reason == "":
            reason = "Model indicates low eligibility score."


    print("\n" + color + f"Applicant is: {final_result}" + Style.RESET_ALL)
    print(Fore.YELLOW + f"Reason: {reason}")
    print(Fore.CYAN + f"Loan-to-Income Ratio: {loan_to_income_ratio}\n")

    result = {
        'Credit_History': credit_history,
        'Income': income,
        'Employment_Type': employment_type,
        'Loan_Amount': loan_amount,
        'Dependents': dependents,
        'Loan_to_Income_Ratio': loan_to_income_ratio,
        'Prediction': final_result,
        'Reason': reason
    }

    file_exists = os.path.isfile('loan_predictions.csv')
    pd.DataFrame([result]).to_csv('loan_predictions.csv', mode='a', header=not file_exists, index=False)

    print(Fore.YELLOW + "\nResult saved to loan_predictions.csv")
    print(Fore.CYAN + "\nThank you for using the Smart Loan Eligibility Predictor!\n")
except ValueError:
    print(Fore.RED + "\n Invalid input! Please enter numeric values correctly." + Style.RESET_ALL)



# ------------------------------------------------------------------------------------------------------
# USING speech_recognition

# import speech_recognition as sr
# import pyttsx3
# import pandas as pd
# from sklearn.tree import DecisionTreeClassifier
# from colorama import Fore, Style, init

# init(autoreset=True)

# print("="*60)
# print(Fore.CYAN + "  Loan Eligibility Prediction System using Decision Tree")
# print(Style.RESET_ALL + "Developed by: " + Fore.YELLOW + " Aditya Sharma Yash Yaduvanshi & Pratham Soni")
# print(Style.RESET_ALL + "Course: " + Fore.MAGENTA + "Data Science (CSE0521) | PBL Project")
# print("="*60 + "\n")

# engine = pyttsx3.init()
# engine.setProperty('rate', 170)

# def speak(text):
#     print("AI:", text)
#     engine.say(text)
#     engine.runAndWait()

# def listen():
#     r = sr.Recognizer()
#     mic_index = 9 
#     try:
#         with sr.Microphone(device_index=mic_index) as source:
#             r.adjust_for_ambient_noise(source, duration=1)
#             speak(" Listening... please speak now.")
#             audio = r.listen(source, timeout=6, phrase_time_limit=6)
#             with open("test_audio.wav", "wb") as f:
#                 f.write(audio.get_wav_data())
#             print(" Audio saved as test_audio.wav")

#         try:
#             text = r.recognize_google(audio)
#             print("You said:", text)
#             return text.lower().strip()
#         except sr.UnknownValueError:
#             speak("Sorry, I could not understand your voice.")
#             return ""
#         except sr.RequestError:
#             speak("Network issue, please check your connection.")
#             return ""
#     except Exception as e:
#         speak(f"Microphone error: {e}")
#         return ""

# # =========================
# #  Train AI Model
# # =========================
# data = {
#     'Credit_History': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
#     'Income': [60000, 30000, 80000, 40000, 20000, 75000, 90000, 25000, 65000, 35000],
#     'Employment_Type': ['salaried', 'self-employed', 'salaried', 'salaried', 'self-employed',
#                         'salaried', 'salaried', 'self-employed', 'salaried', 'self-employed'],
#     'Loan_Amount': [150000, 100000, 250000, 120000, 80000, 200000, 180000, 90000, 160000, 110000],
#     'Dependents': [2, 3, 1, 2, 4, 2, 1, 3, 2, 3],
#     'Loan_Status': ['eligible', 'not eligible', 'eligible', 'eligible', 'not eligible',
#                     'eligible', 'eligible', 'not eligible', 'eligible', 'not eligible']
# }

# df = pd.DataFrame(data)
# df['Employment_Type'] = df['Employment_Type'].map({'salaried': 1, 'self-employed': 0})
# df['Loan_Status'] = df['Loan_Status'].map({'eligible': 1, 'not eligible': 0})

# X = df[['Credit_History', 'Income', 'Employment_Type', 'Loan_Amount', 'Dependents']]
# y = df['Loan_Status']

# model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
# model.fit(X, y)


# speak("Welcome to the AI Loan Eligibility Assistant!")

# # --- Credit History ---
# speak("Please tell your credit history. Say one for good or zero for bad.")
# spoken = listen().replace("one", "1").replace("zero", "0").replace(" ", "")
# if spoken == "":
#     speak("I couldn't understand. Please enter manually — 1 for good, 0 for bad.")
#     spoken = input("Enter manually (1 for good, 0 for bad): ")
# credit = int(spoken[0]) if spoken else 0

# # --- Income ---
# speak("Please tell your monthly income in rupees.")
# income_text = listen()
# if income_text == "":
#     speak("I couldn't catch that. Please type your income manually.")
#     income_text = input("Enter your monthly income (in rupees): ")
# income = int(''.join([c for c in income_text if c.isdigit()]) or "50000")

# # --- Employment Type ---
# speak("Are you salaried or self employed?")
# emp_type_text = listen()
# if emp_type_text == "":
#     speak("Please type if you are salaried or self employed.")
#     emp_type_text = input("Enter your employment type (salaried/self-employed): ")
# emp_type = 1 if "salaried" in emp_type_text.lower() else 0

# # --- Loan Amount ---
# speak("Please tell your desired loan amount.")
# loan_text = listen()
# if loan_text == "":
#     speak("Please type your desired loan amount manually.")
#     loan_text = input("Enter desired loan amount: ")
# loan_amount = int(''.join([c for c in loan_text if c.isdigit()]) or "100000")

# # --- Dependents ---
# speak("How many dependents do you have?")
# dep_text = listen()
# if dep_text == "":
#     speak("Please type number of dependents manually.")
#     dep_text = input("Enter number of dependents: ")
# dependents = int(''.join([c for c in dep_text if c.isdigit()]) or "2")


# sample = pd.DataFrame({
#     'Credit_History': [credit],
#     'Income': [income],
#     'Employment_Type': [emp_type],
#     'Loan_Amount': [loan_amount],
#     'Dependents': [dependents]
# })

# prediction = model.predict(sample)[0]

# if prediction == 1:
#     speak("🎉 Congratulations! You are eligible for the loan.")
# else:
#     speak("❌ Sorry, you are not eligible for the loan at this time.")




