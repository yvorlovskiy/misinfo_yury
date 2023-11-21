import pandas as pd
import httpx
from bs4 import BeautifulSoup
import time
import re
from httpx import ReadTimeout

# Define functions to be used in the script
def remove_after_number(s):
    result = re.match(r".*?\b(\d{4})\b", s)
    return result.group() if result else s
    
def get_date_and_label(content):
    soup = BeautifulSoup(content, 'html.parser')
    dates = soup.findAll("div", {"class": "m-statement__desc"})
    date = dates[0].text.split('on ')[1].split(' in')[0]
    date = remove_after_number(date)
    img_tags = soup.find_all('img')
    label = img_tags[4].get('alt') if len(img_tags) > 4 else "No label found"
    return date, label

def get_article(content):
    soup = BeautifulSoup(content, 'html.parser')

    # First split the content at "Our Sources"
    first_split = str(soup).split('<h3 class="m-superbox__title">Our Sources</h3>', 1)[0]

    split_soup = BeautifulSoup(first_split, 'html.parser')

    # Extract text from all <p> tags
    paragraphs = split_soup.find_all('p')
    text = ' '.join(para.get_text() for para in paragraphs)
    help_phrase = "We need your help."
    if help_phrase in text:
        text = text.split(help_phrase, 1)[1]
    print(text)
    return text


def get_speaker(content):
    soup = BeautifulSoup(content, 'html.parser')

    # Find the <a> tag with class 'm-statement__name'
    speaker_tag = soup.find('a', class_='m-statement__name')
    
    # Extract and return the speaker's name
    if speaker_tag:
        return speaker_tag.get_text(strip=True)
    else:
        return None
# Function to perform the search with retries
def scrape_search(search_id, client, retries=3, delay=1.0):
    url = f"https://www.politifact.com/factchecks/{search_id}"
    for attempt in range(retries):
        try:
            response = client.get(url, timeout=60.0)  # Increase the timeout if needed
            response.raise_for_status()
            return response
        except ReadTimeout:
            print(f"Timeout occurred for ID: {search_id} on attempt {attempt + 1}. Retrying after {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        except httpx.HTTPStatusError as e:
            print(f"Request for ID: {search_id} returned status code {e.response.status_code} on attempt {attempt + 1}")
            return None
    return None  # If all retries failed

# Function to save the DataFrame up to the current index
def save_progress(df, index, id):
    save_path = f"LIAR_new_with_context_{index}_id_{id}_try2.jsonl"
    df.to_json(save_path, orient='records', lines=True)
    print(f"Progress saved at index {index}, ID: {id}")

# Load DataFrame and prepare it for processing
df = pd.read_json('impossible_and_hard_cases.jsonl', lines=True)
#df['example_id'] = df['example_id'].apply(lambda x: x.replace('.json', ''))
df['context'] = ''
df['speaker'] = ''
df_with_old_labels = df.copy()


# Initialize the HTTP client
client = httpx.Client(
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,lt;q=0.8,et;q=0.7,de;q=0.6",
    },
    follow_redirects=True,
    http2=True,
    default_encoding='iso-8859-1',
)


sample = scrape_search('23002', client)
print(get_speaker(sample.content))
last_saved = 0

for index in range(last_saved, len(df)):
    row = df.iloc[index]
    response = scrape_search(row['example_id'], client)
    if response and response.status_code == 200:
        context = get_article(response.content)
        speaker = get_speaker(response.content)
        df.at[index, 'context'] = context
        df.at[index,  'speaker'] = speaker
        print(f"ID: {row['example_id']} - Context: {context[:50]}...")  # Print first 50 characters of context for confirmation
    elif response:
        print(f"Failed to fetch data for ID: {row['example_id']}, status_code: {response.status_code}")
        if response.status_code == 404:
            print(f"No data found for example_id: {row['example_id']} (404 Not Found). Continuing to next example_id.")
            continue
    else:
        print(f"All retries failed for example_id: {row['example_id']}. Saving progress and continuing with the next example_id.")
        save_progress(df, index, row['example_id'])
        continue

    # Save progress every 100 rows or at other intervals as needed
    if index % 100 == 0 and index != last_saved:
        save_progress(df, index, row['example_id'])

    time.sleep(0.5)  # To avoexample_id hitting the server too hard
# Save progress at the end if no errors occurred
if index == len(df) - 1:
    save_progress(df, index, row['example_id'])

# Release the client resources
client.close()
