import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "frontend" / "audio"


STARTER_PHRASES = [
    # Short and natural
    ("starter_1", "One moment please."),
    ("starter_2", "Sure, one moment."),
    ("starter_3", "Got it, hold on."),
    ("starter_4", "On it right now."),
    ("starter_5", "Alright, give me a sec."),
    ("starter_6", "Right, one moment."),
    ("starter_7", "Okay, hold on."),
    ("starter_8", "One second please."),
    ("starter_9", "Give me a moment."),
    ("starter_10", "Just a moment please."),

    # Professional and polite
    ("starter_11", "Certainly, I'll take care of that."),
    ("starter_12", "Understood, working on it."),
    ("starter_13", "Of course, allow me a moment."),
    ("starter_14", "Right away, sir."),
    ("starter_15", "Consider it done, just a moment."),
    ("starter_16", "Absolutely, I'm on it."),
    ("starter_17", "I'll get that ready for you."),
    ("starter_18", "Certainly, processing your request."),
    ("starter_19", "Understood, please bear with me."),
    ("starter_20", "I'll handle that right away."),

    # Intelligent AI assistant style
    ("starter_21", "Processing your request now."),
    ("starter_22", "Let me look into that."),
    ("starter_23", "I'm checking that for you."),
    ("starter_24", "Analyzing the information."),
    ("starter_25", "Working through the details."),
    ("starter_26", "Let me figure that out."),
    ("starter_27", "I'm gathering the necessary information."),
    ("starter_28", "Running a quick analysis."),
    ("starter_29", "Let me process that."),
    ("starter_30", "I'm on it. Just a moment."),

    # Searching and researching
    ("starter_31", "Let me search for that."),
    ("starter_32", "Searching for the information now."),
    ("starter_33", "I'll find that for you."),
    ("starter_34", "Checking a few sources."),
    ("starter_35", "Let me verify that."),
    ("starter_36", "Looking that up right now."),
    ("starter_37", "I'm retrieving the latest information."),
    ("starter_38", "Let me investigate that."),
    ("starter_39", "Searching the available data."),
    ("starter_40", "I'll get you the relevant details."),

    # Calm and conversational
    ("starter_41", "Alright, let's see."),
    ("starter_42", "Just checking that for you."),
    ("starter_43", "Let me have a look."),
    ("starter_44", "Sure, I'll check."),
    ("starter_45", "Alright, I'm looking into it."),
    ("starter_46", "Hang on, I've got this."),
    ("starter_47", "Let's take a look."),
    ("starter_48", "Just give me a little time."),
    ("starter_49", "I'll see what I can find."),
    ("starter_50", "Bear with me for a moment."),

    # Futuristic / JARVIS-style
    ("starter_51", "Systems are processing your request."),
    ("starter_52", "Initiating the requested operation."),
    ("starter_53", "Running diagnostics, one moment."),
    ("starter_54", "Request acknowledged."),
    ("starter_55", "Executing your command."),
    ("starter_56", "Accessing the required information."),
    ("starter_57", "Operation is underway."),
    ("starter_58", "Establishing the necessary connection."),
    ("starter_59", "Analyzing the available data."),
    ("starter_60", "Your request is being processed."),

    # Taking action
    ("starter_61", "Opening that for you now."),
    ("starter_62", "Launching the application."),
    ("starter_63", "Preparing everything now."),
    ("starter_64", "Making the necessary adjustments."),
    ("starter_65", "Setting that up for you."),
    ("starter_66", "Initiating the process."),
    ("starter_67", "Getting everything ready."),
    ("starter_68", "Performing the requested action."),
    ("starter_69", "Working on that now."),
    ("starter_70", "Taking care of it."),

    # Brief acknowledgements
    ("starter_71", "Understood."),
    ("starter_72", "Certainly."),
    ("starter_73", "Absolutely."),
    ("starter_74", "Right away."),
    ("starter_75", "Acknowledged."),
    ("starter_76", "As you wish."),
    ("starter_77", "At once."),
    ("starter_78", "Of course."),
    ("starter_79", "Very well."),
    ("starter_80", "I'm on it."),

    # Thoughtful responses
    ("starter_81", "Let me think that through."),
    ("starter_82", "I'll work out the best approach."),
    ("starter_83", "Let me consider that carefully."),
    ("starter_84", "I'm evaluating the options."),
    ("starter_85", "Let's work through this."),
    ("starter_86", "I'll find the most suitable solution."),
    ("starter_87", "Let me break that down."),
    ("starter_88", "I'm reviewing the details."),
    ("starter_89", "Give me a moment to analyze that."),
    ("starter_90", "Let me put that together for you."),

    # Warm and helpful
    ("starter_91", "Happy to help, one moment."),
    ("starter_92", "Sure thing, I'll take care of it."),
    ("starter_93", "No problem, just a moment."),
    ("starter_94", "Of course, I'm here to help."),
    ("starter_95", "You got it, working on it."),
    ("starter_96", "Absolutely, let me check."),
    ("starter_97", "Leave that to me."),
    ("starter_98", "I'll take it from here."),
    ("starter_99", "Right, let's get that done."),
    ("starter_100", "Certainly, just give me a moment."),
]


PHRASES = STARTER_PHRASES
VOICE = "en-GB-RyanNeural"
RATE = "+15%"


async def generate_one(name: str, text: str) -> bool:
    try:
        import edge_tts

    except ImportError:
        return False

    path = AUDIO_DIR / f"{name}.mp3"

    try:
        communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
        await communicate.save(str(path))
        print(f" [OK] {name}.mp3")
        return True

    except Exception as e:
        print(f" [FAIL] {name}.mp3: {e}")
        return False

async def main():

    try:
        import edge_tts

    except ImportError:
        print("edge-tts not installed. Run: pip install edge-tts")
        return 1

    AUDIO_DIR.mkdir(parents=True,  exist_ok=True)

    for f in AUDIO_DIR.glob("fillowup_*.mp3"):

        try:
            f.unlink()
            print(f" [REMOVED] {f.name}")

        except OSError:
            pass

    print(f"Generating thinking audio in {AUDIO_DIR}...")
    success = 0

    for name, text in PHRASES:
        if await generate_one(name, text):
            success += 1

    print(f"Done: {success}/{len(PHRASES)} files.")
    return 0 if success == len(PHRASES) else 1

if __name__ == "__main__":

    try:
        exit_code = asyncio.run(main())

    except KeyboardInterrupt:
        exit_code = 130

    sys.exit(exit_code)