import requests
import os
import threading
import itertools
import time
import sys
from dotenv import dotenv_values

class LoadingAnimation:
    """
    Creates an animated loading indicator in the terminal while the chatbot processes queries.
    Uses threading to run the animation concurrently with the main process.
    """
    def __init__(self, description="Processing"):
        self.description = description
        self.done = False
        self.spinner = itertools.cycle(['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'])

    def animate(self):
        # Clear the current line to prepare for the animation
        sys.stdout.write('\r')
        while not self.done:
            sys.stdout.write(f'\r{self.description} {next(self.spinner)} ')
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear the animation when done
        sys.stdout.write('\r')
        sys.stdout.flush()

    def start(self):
        # Create and start the animation thread
        self.thread = threading.Thread(target=self.animate)
        self.thread.start()

    def stop(self):
        # Signal the animation to stop and wait for the thread to finish
        self.done = True
        if hasattr(self, 'thread'):
            self.thread.join()

def send_query_with_loading(url, headers, query):
    """
    Sends a query to the chatbot while displaying a loading animation.
    Returns the response from the chatbot.
    """
    # Initialize the loading animation
    loading = LoadingAnimation("Thinking")
    try:
        # Start the loading animation
        loading.start()
        
        # Send the actual query
        response = requests.get(
            url, 
            headers=headers, 
            json={"query": query}
        )
        
        return response
    finally:
        # Ensure the loading animation stops even if there's an error
        loading.stop()

def setup_chatbot_session():
    # Load environment configurations
    config = dotenv_values(".env")
    func_key = config["FUNC_KEY"]
    
    url = "https://func-dataroots-guiden-runquery.azurewebsites.net/api/query_chatbot"
    headers = {
        'x-functions-key': func_key
    }
    return url, headers

def run_interactive_chatbot():
    url, headers = setup_chatbot_session()
    
    print("Welcome to the Interactive Chatbot!")
    print("Type your questions and press Enter. To exit, type 'quit' or 'exit'.")
    print("-" * 50)

    while True:
        user_query = input("\nYour question: ").strip()
        
        if user_query.lower() in ['quit', 'exit']:
            print("\nThank you for using the chatbot. Goodbye!")
            break
            
        if not user_query:
            print("Please type a question!")
            continue
            
        try:
            response = send_query_with_loading(url, headers, user_query)
            
            if response.status_code == 200:
                print("\nChatbot Response:")
                print("-" * 20)
                print(response.text)
                print("-" * 50)
            else:
                print(f"\nError: Failed to get response (Status code: {response.status_code})")
                print(f"Details: {response.content}")
                
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    run_interactive_chatbot()