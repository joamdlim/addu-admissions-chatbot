print("🚀 LLaMA test starting...")

from chatbot.llama_interface import correct_typos, predict_next_words

while True:
    print("\nChoose an option:")
    print("1: Test typo correction")
    print("2: Test next-word prediction")
    print("exit: Quit")
    
    choice = input("Choice: ").lower()
    
    if choice == "exit":
        break
    elif choice == "1":
        user_input = input("💬 Enter text with typos: ")
        corrected = correct_typos(user_input)
        print(f"✨ Corrected: {corrected}")
    elif choice == "2":
        user_input = input("💬 Enter text for next-word prediction: ")
        suggestions = predict_next_words(user_input)
        print("✨ Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
    else:
        print("Invalid choice. Try again.")