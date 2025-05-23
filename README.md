Implemented Components
- NLU (Natural Language Understanding)
- Policy Manager
- NLG (Natural Language Generation)
- 
![image](https://github.com/user-attachments/assets/9b631cb1-7801-4e85-ab6d-10a7f641b703)



Dialogue System Architecture
![image](https://github.com/user-attachments/assets/49bf79a1-1e91-48be-bce1-43bbd91aedd8)


NLU including:
1 - Intent detection
2 - Slot filling
3 - Entity detection

NLG including:
1 - Information Retrieval (getting revenlant information available in documents)
2 - HuggingFace NLG
3 - Gemini NLG (optional)

Policy Manager:
1 - PolicyManager.py (Making sure Slots are filled and intent is detected, then calls NLG to get the answer ...)


Output sample:
![image](https://github.com/user-attachments/assets/ba3b0093-5cb7-4e0e-95f2-f7418ffeaea6)
![image](https://github.com/user-attachments/assets/e0730005-9a86-4163-b935-fee9840d447e)
