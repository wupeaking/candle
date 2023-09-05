import init, { Model } from "./build/m.js";

async function fetchArrayBuffer(url) {
  const res = await fetch(url, {
    cache: "force-cache",
  });
  const data = await res.arrayBuffer();
  return new Uint8Array(data);
}

class Llama2C {
  static instance = {};

  static async getInstance(weightsURL, modelID, tokenizerURL) {
    // load individual modelID only once
    if (!this.instance[modelID]) {
      await init();

      self.postMessage({ status: "loading", message: "Loading Model" });

      const [weightsArrayU8, tokenizerArrayU8] = await Promise.all([
        fetchArrayBuffer(weightsURL),
        fetchArrayBuffer(tokenizerURL),
      ]);

      this.instance[modelID] = new Model(weightsArrayU8, tokenizerArrayU8);
    }
    return this.instance[modelID];
  }
}

let controller = null;
self.addEventListener("message", (event) => {
  if (event.data.command === "start") {
    controller = new AbortController();
    generate(event.data);
  } else if (event.data.command === "abort") {
    controller.abort();
  }
});

async function generate(data) {
  const {
    weightsURL,
    modelID,
    tokenizerURL,
    prompt,
    temp,
    repeatPenalty,
    seed,
    maxSeqLen,
  } = data;
  try {
    self.postMessage({ status: "loading", message: "Starting llama2.c" });
    const model = await Llama2C.getInstance(weightsURL, modelID, tokenizerURL);

    self.postMessage({ status: "loading", message: "Initializing model" });
    model.init_with_prompt(prompt, temp, repeatPenalty, seed);

    const seq_len = model.get_seq_len();

    let sentence = "";
    let max_tokens = maxSeqLen ? maxSeqLen : seq_len - prompt.length - 1;

    while (max_tokens--) {
      await new Promise(async (resolve) => {
        if (controller && controller.signal.aborted) {
          self.postMessage({
            status: "aborted",
            message: "Aborted",
            output: prompt + sentence,
          });
          return;
        }
        const token = await model.next_token();

        sentence += token;
        self.postMessage({
          status: "generating",
          message: "Generating token",
          token: token,
          sentence: sentence,
          prompt: prompt,
        });
        setTimeout(resolve, 0);
      });
    }
    self.postMessage({
      status: "complete",
      message: "complete",
      output: prompt + sentence,
    });
  } catch (e) {
    self.postMessage({ error: e });
  }
}
