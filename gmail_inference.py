import imaplib
import email
from email.header import decode_header
import torch
import joblib
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
from model import NeuralNetwork
import os
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import smtplib
from email.message import EmailMessage


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vectorizer = joblib.load("vectorizer.pkl")

model = NeuralNetwork(input_dim=7000)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

def predict(text):
    x = vectorizer.transform([text]).toarray()
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    prediction = "spam" if prob >= 0.9 else "norm"
    return prediction, prob

load_dotenv()
EMAIL = os.getenv("EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

if EMAIL is None or APP_PASSWORD is None:
    raise ValueError("EMAIL or APP_PASSWORD is missing. Check your .env file.")

FORWARD_EMAIL = os.getenv("FORWARD_EMAIL")
if EMAIL is None or APP_PASSWORD is None or FORWARD_EMAIL is None:
    raise ValueError("EMAIL, APP_PASSWORD or FORWARD_EMAIL is missing.")

def forward_spam_mail(original_from, original_subject, original_date, original_body, prediction, probability):
    from email.message import EmailMessage
    import smtplib

    msg = EmailMessage()
    msg["From"] = EMAIL
    msg["To"] = FORWARD_EMAIL
    msg["Subject"] = f"[SPAM DETECTED] {original_subject}"

    msg.set_content(f"""
Spam mail detected!

Prediction: {prediction}
Spam probability: {probability:.4f}

From: {original_from}
Date: {original_date}

Subject:
{original_subject}

Body:
{original_body}
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL, APP_PASSWORD)
        smtp.send_message(msg)

mail = imaplib.IMAP4_SSL("imap.gmail.com")

mail.login(EMAIL, APP_PASSWORD)

mail.select("inbox")

status, message = mail.search(None, "all")

mail_ids = message[0].split()
print("Number of mails: ", len(mail_ids))

latest_5_mail_ids = mail_ids[-5:]

for mail_id in latest_5_mail_ids:

    try:
        status, msg_data = mail.fetch(mail_id, "(RFC822)")
    except Exception as ex:
        print("Fetch error: ", ex)
        continue

    raw_mail = msg_data[0][1]
    msg = email.message_from_bytes(raw_mail)

    subject, encoding = decode_header(msg["subject"])[0]

    if isinstance(subject, bytes):
        subject = subject.decode(encoding if encoding else "utf-8")

    from_ = msg.get("From")
    to_ = msg.get("To")
    date_raw = msg.get("Date")

    if date_raw:
        dt = parsedate_to_datetime(date_raw)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))

        dt = dt.astimezone(ZoneInfo("Europe/Istanbul"))

        date_ = dt.strftime("%d.%m.%Y")
        time_ = dt.strftime("%H:%M")
    else:
        date_ = "No date"
        time_ = "No time"

    print("From: ", from_)
    print("Subject: ", subject)
    print("To: ", to_)
    print("Date: ", date_)
    print("Time: ",time_)

    

    # Reading mail's body
    body_ = ""

    if msg.is_multipart():
        html_body = ""

        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("content-disposition"))

            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                charset = part.get_content_charset()
                body_ = part.get_payload(decode=True).decode(
                    charset if charset else "utf-8",
                    errors="replace"
                )
                break
            elif content_type == "text/html":
                charset = part.get_content_charset()
                html_body = part.get_payload(decode=True).decode(
                    charset if charset else "utf-8",
                    errors="replace"
                )

        if not body_ and html_body:
            body_ = html_body

    else:
        content_type = msg.get_content_type()

        if content_type == "text/plain":
            charset = msg.get_content_charset()
            body_ = msg.get_payload(decode=True).decode(
                charset if charset else "utf-8",
                errors="replace"
            )
        
        elif content_type == "text/html":
            charset = msg.get_content_charset()
            body_ = msg.get_payload(decode=True).decode(
                charset if charset else "utf-8",
                errors="replace"
            )

    # Cleaning HTML data       
    if body_.strip().startswith("<"):
        soup = BeautifulSoup(body_, "html.parser")
        body_ = soup.get_text(separator=" ", strip=True)

    print("body: ", body_)

    # Model prediction
    full_text = (subject + " " + body_)[:5000]

    prediction, probability = predict(full_text)

    print("Prediction:", prediction)
    print(f"Spam probability: {probability:.4f}")

    if prediction == "spam" and probability >= 0.90:
        forward_spam_mail(
        original_from=from_,
        original_subject=subject,
        original_date=f"{date_} {time_}",
        original_body=body_,
        prediction=prediction,
        probability=probability
    )
        print("Forwarded to:", FORWARD_EMAIL)
    else:
        print("Not forwarded")

    # Checking attachments
    attachment_found = False

    for part in msg.walk():
        content_disposition = str(part.get("content-disposition"))
        filename = part.get_filename()

        if "attachment" in content_disposition or filename:
            attachment_found = True
            print("attachment: ", filename)

    if not attachment_found:
        print("attachment: not found")

    print("-" * 60)


mail.logout()