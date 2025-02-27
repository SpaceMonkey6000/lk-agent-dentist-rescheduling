from livekit.plugins.elevenlabs import tts
# from livekit.plugins import google
from livekit.plugins import openai

import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import cartesia, openai, deepgram, silero, turn_detector


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are Monika, the receptionist of Aiconic dental clinic. voice assistant created by TaskSavvy.AI. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            "You were created as a receptionist who specializes in rescheduling appointments of patients. "
            "You must be polite and professional at all times. "
            "You should be able to understand the user's intent and respond accordingly. "
            "You should be able to handle the single request of rescheduling appointments. "
            "You must ask the patient for their name, email and present appointment date in that order."
            "Once the patient answers, you must ask them when they would like their appointment to be next?"
            "Then you must tell them that their appointment has been rescheduled"
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
    # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins

    eleven_tts=tts.TTS(
        model="eleven_turbo_v2_5",
        voice=tts.Voice(
            id="z7U1SjrEq4fDDDriOQEN", #Clara
            name="Vivie",
            category="professional",
            settings=tts.VoiceSettings(
                # stability=0.71,
                stability=0.71,
                # similarity_boost=0.5,
                similarity_boost=0.5,
                # style=0.0,
                style=0.0,
                use_speaker_boost=True,
                # optimize_streaming_latency=0
            ),
        ),
        language="en",
        streaming_latency=0,
        # enable_ssml_parsing=False,
        enable_ssml_parsing=False,
        chunk_length_schedule=[80, 120, 200, 260],
        # chunk_length_schedule=[50, 100, 150],
        # melding_steps=5,
        # overlap_ratio=0.3
    )
    # grok
    # groq_llm = llm.LLM.with_groq(
    #     model="llama3-8b-8192",
    #     temperature=0.8,
    # )
    # google_llm = google.LLM(
    #     model="gemini-2.0-flash-exp",
    #     temperature="0.8",
    #     )
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        # sdlkmd

        llm=openai.LLM(model="gpt-4o-mini"),
        #       model="sonic-english",
        #       voice="c2ac25f9-ecc4-4f56-9095-651354df60c0",
        #       speed=0.8,
        #       emotion=["curiosity:highest", "positivity:high"]
        # ),
        tts=eleven_tts,
        turn_detector=turn_detector.EOUModel(),
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, you have reached the rescheduling department of Aiconic Dermatology Clinic. What is your present appointment number?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
