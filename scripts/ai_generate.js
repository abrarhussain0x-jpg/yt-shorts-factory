#!/usr/bin/env node
/**
 * scripts/ai_generate.js — Node.js helper script for AI-powered metadata generation.
 *
 * Called by core/ai_metadata.py via subprocess. Uses z-ai-web-dev-sdk to generate
 * viral titles, descriptions, hashtags, and captions for YouTube Shorts, TikTok,
 * and Instagram Reels.
 *
 * Usage:
 *   node scripts/ai_generate.js --prompt "..."
 *
 * Outputs JSON to stdout. Handles errors gracefully and always outputs valid JSON
 * (either the result or an error object).
 */

"use strict";

// ── Argument Parsing ──────────────────────────────────────────

function parseArgs() {
  const args = process.argv.slice(2);
  let prompt = "";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--prompt" && i + 1 < args.length) {
      prompt = args[i + 1];
      i++; // skip the value
    }
  }

  return { prompt };
}

// ── Safe JSON Output ──────────────────────────────────────────

function outputResult(data) {
  process.stdout.write(JSON.stringify(data) + "\n");
}

function outputError(message, details) {
  outputResult({
    error: message,
    details: details || null,
  });
}

// ── Main ──────────────────────────────────────────────────────

async function main() {
  const { prompt } = parseArgs();

  if (!prompt) {
    outputError("Missing required argument: --prompt");
    process.exit(1);
  }

  if (prompt.length > 16000) {
    outputError("Prompt too long (max 16000 characters)", {
      length: prompt.length,
    });
    process.exit(1);
  }

  let zai;
  try {
    const ZAI = (await import("z-ai-web-dev-sdk")).default;
    zai = await ZAI.create();
  } catch (err) {
    outputError("Failed to initialize z-ai-web-dev-sdk", {
      message: err.message,
    });
    process.exit(1);
  }

  try {
    const completion = await zai.chat.completions.create({
      messages: [
        {
          role: "system",
          content:
            "You are a social media metadata expert. Always respond with valid JSON only. " +
            "No markdown, no code fences, no explanation outside the JSON.",
        },
        { role: "user", content: prompt },
      ],
    });

    // Extract the text content from the completion response
    let text = "";
    if (completion && completion.choices && completion.choices.length > 0) {
      const choice = completion.choices[0];
      text = choice.message?.content || choice.text || "";
    } else if (typeof completion === "string") {
      text = completion;
    } else if (completion && completion.content) {
      text = completion.content;
    }

    if (!text) {
      outputError("Empty response from AI", { completion: String(completion) });
      process.exit(1);
    }

    // Try to parse the AI response as JSON
    // The AI might wrap JSON in code fences — strip them
    let cleanedText = text.trim();
    if (cleanedText.startsWith("```")) {
      cleanedText = cleanedText.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "");
    }

    let parsed;
    try {
      parsed = JSON.parse(cleanedText);
    } catch {
      // If parsing fails, return the raw text so the Python side can handle it
      outputResult({
        raw_text: cleanedText,
        parsed: false,
      });
      return;
    }

    outputResult({
      data: parsed,
      parsed: true,
    });
  } catch (err) {
    outputError("AI completion request failed", {
      message: err.message,
      name: err.name || "UnknownError",
    });
    process.exit(1);
  }
}

main().catch((err) => {
  outputError("Unexpected error in ai_generate.js", {
    message: err.message,
  });
  process.exit(1);
});
