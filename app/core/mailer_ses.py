import os
import smtplib
import ssl
from email.message import EmailMessage

GMAIL_USER = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
VERIFY_TOKEN_TTL_MINUTES = int(os.getenv("VERIFY_TOKEN_TTL_MINUTES", "60"))
RESET_TOKEN_TTL_MINUTES = int(os.getenv("RESET_TOKEN_TTL_MINUTES", "30"))

class GmailMailer:
    """
    Simple Gmail SMTP sender using an App Password.
    """
    def send_password_reset(self, to_email: str, reset_link: str) -> None:
        if not GMAIL_USER or not GMAIL_APP_PASSWORD:
            # Don’t raise to avoid leaking infra details; just no-op in dev misconfig
            return

        msg = EmailMessage()
        msg["Subject"] = "Reset your password for TradeMind"
        msg["From"] = f"TradeMind <{GMAIL_USER}>"
        msg["To"] = to_email
        msg.set_content(
            f"Reset your TradeMind password\n\n"
            f"You recently requested to reset your password. This link expires in {RESET_TOKEN_TTL_MINUTES} minutes.\n\n"
            f"Reset link:\n{reset_link}\n\n"
            f"If you didn’t request this, you can safely ignore this email."
        )
        msg.add_alternative(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width,initial-scale=1" />
          <title>Reset your password</title>
        </head>
        <body style="margin:0;padding:0;background-color:#f6f7fb;">
          <span style="display:none !important;visibility:hidden;opacity:0;color:transparent;height:0;width:0;overflow:hidden;">
            Reset your TradeMind password. This link expires in {RESET_TOKEN_TTL_MINUTES} minutes.
          </span>
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f6f7fb;">
            <tr>
              <td align="center" style="padding:24px;">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="width:100%;max-width:600px;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;">
                  <tr>
                    <td style="padding:24px 24px 0 24px;background:#111827;">
                      <h1 style="margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:20px;line-height:28px;color:#ffffff;">
                        TradeMind
                      </h1>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:24px;">
                      <h2 style="margin:0 0 8px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:22px;line-height:28px;color:#111827;">
                        Reset your password
                      </h2>
                      <p style="margin:0 0 16px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:16px;line-height:24px;color:#374151;">
                        We received a request to reset your password. Click the button below to choose a new one.
                      </p>
                      <div style="text-align:center;margin:28px 0;">
                        <a href="{reset_link}" target="_blank" rel="noopener noreferrer"
                           style="display:inline-block;background-color:#2563eb;color:#ffffff;text-decoration:none;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:16px;line-height:20px;padding:12px 20px;border-radius:8px;">
                          Reset password
                        </a>
                      </div>
                      <p style="margin:0 0 8px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:22px;color:#6b7280;">
                        This link expires in {RESET_TOKEN_TTL_MINUTES} minutes.
                      </p>
                      <p style="margin:0 0 16px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:22px;color:#6b7280;">
                        If the button doesn’t work, copy and paste this URL into your browser:
                      </p>
                      <p style="margin:0 0 16px 0;word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,'Liberation Mono','Courier New',monospace;font-size:13px;line-height:20px;color:#111827;">
                        {reset_link}
                      </p>
                      <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;" />
                      <p style="margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:12px;line-height:20px;color:#9ca3af;">
                        If you didn’t request a password reset, you can safely ignore this email or contact support if you have concerns.
                      </p>
                    </td>
                  </tr>
                </table>
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:12px;line-height:20px;color:#9ca3af;margin-top:12px;">
                  TradeMind
                </div>
              </td>
            </tr>
          </table>
        </body>
        </html>""", subtype="html")

        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls(context=context)
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.send_message(msg)

    def send_email_verification(self, to_email: str, verify_link: str) -> None:
        if not GMAIL_USER or not GMAIL_APP_PASSWORD:
            return
        msg = EmailMessage()
        msg["Subject"] = "Verify your email for TradeMind"
        msg["From"] = f"TradeMind <{GMAIL_USER}>"
        msg["To"] = to_email
        msg.set_content(
            f"Verify your TradeMind email address\n\n"
            f"Confirm your email to finish setting up your TradeMind account.\n\n"
            f"Verification link (expires in {VERIFY_TOKEN_TTL_MINUTES} minutes):\n{verify_link}\n\n"
            f"If you didn't request this, you can safely ignore this email."
        )
        msg.add_alternative(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8" />
          <meta name="viewport" content="width=device-width,initial-scale=1" />
          <title>Verify your email</title>
        </head>
        <body style="margin:0;padding:0;background-color:#f6f7fb;">
          <span style="display:none !important;visibility:hidden;opacity:0;color:transparent;height:0;width:0;overflow:hidden;">
            Confirm your email to finish setting up your TradeMind account. This link expires in {VERIFY_TOKEN_TTL_MINUTES} minutes.
          </span>
          <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f6f7fb;">
            <tr>
              <td align="center" style="padding:24px;">
                <table role="presentation" width="600" cellspacing="0" cellpadding="0" style="width:100%;max-width:600px;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;">
                  <tr>
                    <td style="padding:24px 24px 0 24px;background:#111827;">
                      <h1 style="margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:20px;line-height:28px;color:#ffffff;">
                        TradeMind
                      </h1>
                    </td>
                  </tr>
                  <tr>
                    <td style="padding:24px;">
                      <h2 style="margin:0 0 8px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:22px;line-height:28px;color:#111827;">
                        Verify your email
                      </h2>
                      <p style="margin:0 0 16px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:16px;line-height:24px;color:#374151;">
                        Confirm your email to finish setting up your TradeMind account.
                      </p>
                      <div style="text-align:center;margin:28px 0;">
                        <a href="{verify_link}" target="_blank" rel="noopener noreferrer"
                           style="display:inline-block;background-color:#2563eb;color:#ffffff;text-decoration:none;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:16px;line-height:20px;padding:12px 20px;border-radius:8px;">
                          Verify email
                        </a>
                      </div>
                      <p style="margin:0 0 8px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:22px;color:#6b7280;">
                        This link expires in {VERIFY_TOKEN_TTL_MINUTES} minutes.
                      </p>
                      <p style="margin:0 0 16px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:22px;color:#6b7280;">
                        If the button doesn't work, copy and paste this URL into your browser:
                      </p>
                      <p style="margin:0 0 16px 0;word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,'Liberation Mono','Courier New',monospace;font-size:13px;line-height:20px;color:#111827;">
                        {verify_link}
                      </p>
                      <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0;" />
                      <p style="margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:12px;line-height:20px;color:#9ca3af;">
                        You're receiving this email because someone signed up for TradeMind with this address. If you didn't request this, you can safely ignore this email.
                      </p>
                    </td>
                  </tr>
                </table>
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:12px;line-height:20px;color:#9ca3af;margin-top:12px;">
                  TradeMind
                </div>
              </td>
            </tr>
          </table>
        </body>
        </html>""", subtype="html")

        ctx = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as s:
            s.starttls(context=ctx)
            s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            s.send_message(msg)
