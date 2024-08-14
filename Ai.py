import os
import pyaudio
import wave
from openai import OpenAI
from google.cloud import speech_v1p1beta1
from google.cloud import texttospeech
from langdetect import detect
import io
from pydub import AudioSegment
from pydub.playback import play
import webbrowser

# 구글 클라우드 API 키 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "InPut Google Cloude API json name"

# OpenAI API 키 설정
client = OpenAI(api_key="Input OpenAI API")


def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("음성을 입력 받고 있습니다.")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("음성 입력 완료하였습니다.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    with open(WAVE_OUTPUT_FILENAME, 'rb') as f:
        audio_data = f.read()

    return audio_data


def transcribe_audio(audio_data):
    client = speech_v1p1beta1.SpeechClient()
    audio = speech_v1p1beta1.RecognitionAudio(content=audio_data)

    # 설정 가능한 언어 목록
    language_codes = ["ko-KR"]
    transcriptions = {}

    # 각 언어에 대해 음성 인식 수행
    for language_code in language_codes:
        config = speech_v1p1beta1.RecognitionConfig(
            encoding=speech_v1p1beta1.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=language_code
        )
        response = client.recognize(config=config, audio=audio)
        if response.results:
            transcriptions[language_code] = response.results[0].alternatives[0].transcript

    # 가장 긴 텍스트를 선택하여 언어를 감지
    detected_language_code = max(transcriptions, key=lambda k: len(
        transcriptions[k])) if transcriptions else None
    return transcriptions.get(detected_language_code, "")


def synthesize_text(text, output_file, voice_gender=texttospeech.SsmlVoiceGender.MALE):
    client = texttospeech.TextToSpeechClient()

    detected_language = detect(text)
    language_code = "ko-KR" if detected_language == "ko" else "en-US"

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=voice_gender
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(output_file, "wb") as out:
        out.write(response.audio_content)

    print(f'Audio content written to file {output_file}')


def play_audio(audio_data, speed_factor=1.0):
    with io.BytesIO(audio_data) as wav_buffer:
        wav_buffer.seek(0)
        audio = AudioSegment.from_file(wav_buffer, format="wav")
        adjusted_audio = audio.speedup(playback_speed=speed_factor)
        play(adjusted_audio)


def generate_response(messages, user_input):
    messages.append({"role": "user", "content": user_input})
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    return response


def main():
    print("가동 시작")

    welcome_message = "안녕하세요 Ai상담사 입니다. 어떤게 궁금하신가요?"
    welcome_output_file = "welcome.wav"
    synthesize_text(welcome_message, welcome_output_file)
    with open(welcome_output_file, "rb") as f:
        welcome_audio_data = f.read()
    play_audio(welcome_audio_data, speed_factor=1.1)

    conversation_history = [
        {"role": "system", "content": "너의 이름은 바디브이고 고객에게 도움을 주는 AI 챗봇 도우미야."},
        {"role": "assistant", "content": "안녕하세요 Ai상담사 바디브 입니다. 궁금하신 점이 있으시면 말씀해주세요"},
        {"role": "user", "content": "배송 방법에 대해 알려줘"},
        {"role": "assistant", "content": "저희 사이트의 모든 제품은 신선한 상태로 안전하게 포장되어 배송됩니다. 주문 후 평균 1-3일 내에 배송이 완료되며, 자세한 배송 상황은 주문 확인 이메일을 통해 확인하실 수 있습니다. 추가 문의 사항이 있으시면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "배송 상태를 알고 싶어"},
        {"role": "assistant", "content": "저희 사이트의 모든 제품은 신선한 상태로 안전하게 포장되어 배송됩니다. 주문 후 평균 1-3일 내에 배송이 완료되며, 자세한 배송 상황은 주문 확인 이메일을 통해 확인하실 수 있습니다. 추가 문의 사항이 있으시면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "배송 오류가 났어"},
        {"role": "assistant", "content": "배송 과정에서 오류가 발생한 경우, 신선한 상태로 안전하게 포장되어 배송되었는지 확인해 주세요. 그래도 문제가 해결되지 않으면 고객 서비스 센터로 연락해 주세요. 추가 문의 사항이 있으시면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "배송 취소하고 싶어"},
        {"role": "assistant", "content": "배송 취소를 원하시면 고객 서비스 센터로 연락해 주세요. 저희 사이트에서는 배송 취소 절차는 고객 서비스 센터를 통해 진행됩니다. 추가 문의 사항이 있으시면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "배송은 어떤 방식으로 진행돼?"},
        {"role": "assistant", "content": "저희 사이트의 모든 제품은 신선한 상태로 안전하게 포장되어 배송됩니다. 주문 후 평균 1-3일 내에 배송이 완료되며, 자세한 배송 상황은 주문 확인 이메일을 통해 확인하실 수 있습니다. 추가 문의 사항이 있으시면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "배송 내역을 확인하고 싶어"},
        {"role": "assistant", "content": "저희 사이트의 모든 제품은 신선한 상태로 안전하게 포장되어 배송됩니다. 배송 내역은 주문 내역 페이지에서 확인하실 수 있으며, 추가 문의 사항이 있으시면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "배송 문의는 어디로 해야 해?"},
        {"role": "assistant", "content": "배송 관련 문의는 고객 서비스 센터로 연락해 주세요. 저희 서비스 이용 중 궁금한 점이나 문제가 발생하면 언제든지 고객 서비스 센터에서 도와드리겠습니다. 추가 문의 사항이 있으시면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "결제 방법에 대해 알려줘"},
        {"role": "assistant", "content": "저희 사이트에서는 카드결제와 계좌이체 결제 방법을 지원합니다. 안전한 결제를 위해 모든 거래는 암호화됩니다. 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "결제 상태를 알고 싶어"},
        {"role": "assistant",
            "content": "저희 사이트에서는 카드결제와 계좌이체 결제 방법을 지원합니다. 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "결제 오류가 났어"},
        {"role": "assistant", "content": "결제 과정에서 오류가 발생한 경우, 카드결제와 계좌이체 결제 방법을 다시 확인해 주세요. 그래도 문제가 해결되지 않으면 고객 서비스 센터로 연락해 주세요. 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "결제 취소하고 싶어"},
        {"role": "assistant", "content": "결제 취소를 원하시면 고객 서비스 센터로 연락해 주세요. 저희 사이트에서는 카드결제와 계좌이체 결제 방법을 지원하며, 결제 취소 절차는 고객 서비스 센터를 통해 진행됩니다. 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "결제 내역을 확인하고 싶어"},
        {"role": "assistant", "content": "저희 사이트에서는 카드결제와 계좌이체 결제 방법을 지원합니다. 결제 내역은 주문 내역 페이지에서 확인하실 수 있으며, 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "결제 보안은 어떻게 돼?"},
        {"role": "assistant", "content": "저희 사이트에서는 카드결제와 계좌이체 결제 방법을 지원하며, 모든 거래는 암호화되어 안전하게 처리됩니다. 고객님의 결제 정보는 안전하게 보호됩니다. 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "결제는 어떤 방식으로 진행돼?"},
        {"role": "assistant",
            "content": "저희 사이트에서는 카드결제와 계좌이체 결제 방법을 지원합니다. 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "결제 문의는 어디로 해야 해?"},
        {"role": "assistant", "content": "결제 관련 문의는 고객 서비스 센터로 연락해 주세요. 저희 사이트에서는 카드결제와 계좌이체 결제 방법을 지원하며, 모든 결제 관련 문의는 고객 서비스 센터에서 도와드리겠습니다. 자세한 내용은 결제 페이지를 참고해 주세요."},
        {"role": "user", "content": "로그인 방법에 대해 알려줘"},
        {"role": "assistant", "content": "저희 사이트에 로그인하시려면, 상단 메뉴의 '로그인' 버튼을 클릭한 후 아이디와 비밀번호를 입력해 주세요. 로그인 후 다양한 서비스를 이용하실 수 있습니다. 문제가 발생하면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "회원가입은 어떻게 해?"},
        {"role": "assistant", "content": "회원가입을 하시려면, 상단 메뉴의 '회원가입' 버튼을 클릭한 후 필요한 정보를 입력해 주세요. 가입이 완료되면 다양한 혜택과 서비스를 이용하실 수 있습니다. 문제가 발생하면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "회원정보를 수정하려면 어떻게 해?"},
        {"role": "assistant", "content": "회원정보를 수정하시려면, 로그인 후 '마이페이지'에서 '프로필 편집'을 클릭해 변경할 정보를 입력해 주세요. 문제가 발생하면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "이메일 주소를 변경하고 싶어"},
        {"role": "assistant", "content": "이메일 주소를 변경하시려면, 로그인 후 '마이페이지'에서 '프로필 편집'을 클릭해 이메일 주소를 변경해 주세요. 문제가 발생하면 고객 서비스 센터로 연락해 주세요."},
        {"role": "user", "content": "내 정보는 어떻게 관리돼?"},
        {"role": "assistant", "content": "회원님의 정보는 저희 사이트에서 안전하게 관리됩니다. 개인정보 보호를 위해 모든 데이터는 암호화되어 저장되며, 철저한 보안 시스템을 통해 보호됩니다. 자세한 내용은 개인정보 처리방침을 참고해 주세요."},
        {"role": "user", "content": "회원 관련 문의는 어디로 해야 해?"},
        {"role": "assistant", "content": "회원 관련 문의는 고객 서비스 센터로 연락해 주세요. 저희 서비스 이용 중 궁금한 점이나 문제가 발생하면 언제든지 고객 서비스 센터에서 도와드리겠습니다."}
    ]

    seafood_keywords = ["고등어", "랍스타", "새우", "참치", "연어",
                        "게", "조개", "전복", "굴", "오징어", "문어", "멍게", "소라"]

    while True:
        audio_data = record_audio()
        user_input = transcribe_audio(audio_data)
        print("사용자: ", user_input)

        if user_input == "전원 종료":
            shutdown_message = "알겠습니다. 좋은 하루 되세요."
            shutdown_output_file = "shutdown.wav"
            synthesize_text(shutdown_message, shutdown_output_file)
            with open(shutdown_output_file, "rb") as f:
                shutdown_audio_data = f.read()
            play_audio(shutdown_audio_data, speed_factor=1.1)
            break
        elif "상품 리스트" in user_input or "상품 모음" in user_input:
            response = "네 알겠습니다. 상품리스트 페이지로 이동합니다."
            webbrowser.open("-")
            response = "이동이 완료되었습니다. Ai를 종료합니다."
            synthesize_text(response, "response.wav")
            with open("response.wav", "rb") as f:
                audio_data = f.read()
            play_audio(audio_data, speed_factor=1.1)
            break
        elif "로그인" in user_input:
            response = "네 알겠습니다. 로그인 페이지로 이동합니다."
            webbrowser.open("-")
            response = "이동이 완료되었습니다. Ai를 종료합니다."
            synthesize_text(response, "response.wav")
            with open("response.wav", "rb") as f:
                audio_data = f.read()
            play_audio(audio_data, speed_factor=1.1)
            break
        elif "주문 관리" in user_input or "주문관리" in user_input:
            response = "네 알겠습니다. 주문관리로 이동합니다."
            webbrowser.open("-")
            response = "이동이 완료되었습니다. Ai를 종료합니다."
            synthesize_text(response, "response.wav")
            with open("response.wav", "rb") as f:
                audio_data = f.read()
            play_audio(audio_data, speed_factor=1.1)
            break
        elif "고객 센터" in user_input or "고객센터" in user_input:
            response = "네 알겠습니다. 고객센터로 이동합니다."
            webbrowser.open("-")
            response = "이동이 완료되었습니다. Ai를 종료합니다."
            synthesize_text(response, "response.wav")
            with open("response.wav", "rb") as f:
                audio_data = f.read()
            play_audio(audio_data, speed_factor=1.1)
            break
        elif user_input:
            # 수산물 관련 질문 또는 conversation_history에 포함된 키워드만 GPT로 전달
            if any(keyword in user_input for keyword in seafood_keywords) or any(keyword in user_input for keyword in ["배송", "결제", "회원", "로그인"]):
                response = generate_response(conversation_history, user_input)
                print("AI : ", response)
                ai_output_file = "ai_response.wav"
                synthesize_text(response, ai_output_file)
                with open(ai_output_file, "rb") as f:
                    ai_audio_data = f.read()
                play_audio(ai_audio_data, speed_factor=1.1)
            else:
                response = "그 질문은 답변드릴 수 없습니다. 저는 홈페이지 전용 AI상담사입니다. 수산물 또는 홈페이지 이용에 관련된 질문만 해주세요."
                synthesize_text(response, "response.wav")
                with open("response.wav", "rb") as f:
                    audio_data = f.read()
                play_audio(audio_data, speed_factor=1.1)
        else:
            wait_message = "현재 대기 중 입니다."
            wait_output_file = "wait.wav"
            synthesize_text(wait_message, wait_output_file)
            with open(wait_output_file, "rb") as f:
                wait_audio_data = f.read()
            play_audio(wait_audio_data, speed_factor=1.1)


if __name__ == "__main__":
    main()
