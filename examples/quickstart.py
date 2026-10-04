"""Minimal usage. Needs TYPESAFE_API_KEY in the environment."""
from jev_saver import JevSaver

labels = {
    "billing": "Payment, invoice or subscription issues",
    "technical": "Bugs, errors or integration problems",
    "account": "Login, password or profile changes",
}
saver = JevSaver(labels, "Which team should handle this ticket?", model="jev-1.13.0")

for msg in ["I was charged twice this month",
            "I was charged twice this month",          # repeated -> answered locally
            "The app crashes when I open settings"]:
    d = saver.decide(msg)
    print(f"{d.source:5}  {d.choice:10}  conf={d.confidence:.2f}  {d.ms:.0f} ms  | {msg}")

print(f"saved {saver.saved_fraction():.0%} of Jev calls")
saver.save("memory.npz")      # reuse later with saver.load("memory.npz")
