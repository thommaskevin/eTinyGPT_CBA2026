#include "model_emb12_h2_l1_d001_ep300_bs4_lr001_model.h"

using namespace model_emb12_h2_l1_d001_ep300_bs4_lr001;

// =======================================================
// CONFIGURAÇÕES
// =======================================================

#define ANSWER_BUFFER_SIZE 256

char answerBuffer[ANSWER_BUFFER_SIZE];

unsigned long start_time = 0;
unsigned long end_time = 0;
unsigned long width_time = 0;

// Pergunta padrão
const String defaultQuestion =
  "How old are moped riders typically?";

// Buffer para entrada via Serial
String buffer = "";

// Número máximo de tokens gerados
int maxNewTokensForAnswer = 10;


// =======================================================
// SETUP
// =======================================================

void setup() {

  Serial.begin(115200);

  // Pequena espera para inicialização da Serial
  delay(1000);

  Serial.println();
  Serial.println("========================================");
  Serial.println("      ESP32 TinyGPT - Serial Q&A");
  Serial.println("========================================");
  Serial.println();

  Serial.println("Serial communication ready.");
  Serial.println("Baud rate: 115200");
  Serial.println();

  Serial.println("Enter your question and press ENTER.");
  Serial.println();

  Serial.print("Example: ");
  Serial.println(defaultQuestion);

  Serial.println();
  Serial.println("Next question:");
  Serial.println();
}


// =======================================================
// PROCESSAR PERGUNTA
// =======================================================

void processQuestion(const String& question) {

  Serial.println();
  Serial.println("----------------------------------------");

  Serial.print("Question: ");
  Serial.println(question);

  Serial.println("----------------------------------------");

  Serial.println("Generating answer...");

  // -----------------------------------------------------
  // Início da medição
  // -----------------------------------------------------

  start_time = millis();

  // Limpa o buffer antes da geração
  memset(answerBuffer, 0, ANSWER_BUFFER_SIZE);

  // -----------------------------------------------------
  // Geração da resposta
  // -----------------------------------------------------

  generate_qna_answer(
    answerBuffer,
    ANSWER_BUFFER_SIZE,
    question,
    maxNewTokensForAnswer
  );

  // -----------------------------------------------------
  // Fim da medição
  // -----------------------------------------------------

  end_time = millis();

  width_time = end_time - start_time;

  // -----------------------------------------------------
  // Exibir resposta
  // -----------------------------------------------------

  Serial.print("Answer: ");
  Serial.println(answerBuffer);

  // -----------------------------------------------------
  // Exibir tempo
  // -----------------------------------------------------

  Serial.print("Processing time (ms): ");
  Serial.println(width_time);

  Serial.println("----------------------------------------");
  Serial.println();

  Serial.println("Next question:");
  Serial.println();
}


// =======================================================
// LOOP
// =======================================================

void loop() {

  // =====================================================
  // RECEBER PERGUNTA VIA SERIAL
  // =====================================================

  while (Serial.available()) {

    char c = Serial.read();

    // ---------------------------------------------------
    // Final da mensagem
    // ---------------------------------------------------

    if (c == '\n' || c == '\r') {

      // Remove espaços e caracteres extras
      buffer.trim();

      // -------------------------------------------------
      // Se existe uma pergunta, processa
      // -------------------------------------------------

      if (buffer.length() > 0) {

        processQuestion(buffer);

      }

      // Limpa o buffer
      buffer = "";
    }

    // ---------------------------------------------------
    // Adiciona caractere ao buffer
    // ---------------------------------------------------

    else {

      buffer += c;
    }
  }
}