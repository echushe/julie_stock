import os

# Print terminal hints for the user to input text containing ticker information:
def print_terminal_hints():
    print("Please input text containing ticker information.")
    print("Example: 'AAPL, MSFT, GOOGL' or 'Apple Inc., Microsoft Corp., Alphabet Inc.'")
    print("Type 'exit' to quit.")


def distill_ashare_ticker_ids_from_text(text):
    """
    Extracts ticker IDs from a given text input.
    
    Args:
        text (str): The input text containing ticker information.
        
    Returns:
        list: A list of extracted ticker IDs.
    """
    # For simplicity, let's assume the tickers are separated by commas
    text_list = [ticker.strip(':').strip('\'').strip('\"').upper() for ticker in text.split()]
    tickers = set()
    for item in text_list:
        # The item should be 6 characters long and consist of digits
        if len(item) != 6 or not item.isdigit():
            continue

        tickers.add(item)

    # Here you can add logic to validate or process the tickers if needed
    return sorted(list(tickers))


# Wait for user input and extract tickers from the text:
def extract_tickers_from_text():

    text = ''
    print('Please enter the text containing ticker information (type "exit" to quit):')
    while True:
        user_input = input()
        text += user_input + ' '
        if user_input.lower() == 'end_of_text':
            break

    tickers = distill_ashare_ticker_ids_from_text(text)
    print("Extracted Tickers:", tickers)
    return tickers
    

if __name__ == "__main__":
    print_terminal_hints()
    tickers = extract_tickers_from_text()
    
    if tickers:
        print("You entered the following tickers:", tickers)
    else:
        print("No tickers were extracted.")
    
    # Here you can add further processing of the extracted tickers if needed
    # For example, saving them to a file or using them in another part of your application
