import dotenv from "dotenv";
dotenv.config({ override: true });
import express from "express";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import path from "path";
import fs from "fs";

async function startServer() {
  const app = express();
  const PORT = 3000;
  
  app.use(express.json());

  // Initialize Gemini if key exists
  const ai = process.env.GEMINI_API_KEY ? new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }) : null;

  // ASI Nodes representing Google Labs Experiments
  let ASINodes = [
    { id: "astra", name: "Project Astra", role: "Real-time Multimodal Perception", recent_activity: 100 },
    { id: "notebooklm", name: "NotebookLM", role: "Source-Grounded Semantic Synthesis", recent_activity: 100 },
  ];

  // SSE Endpoint for telemetry streaming (Simulating Akkurat/AtomTN runtime)
  app.get("/api/telemetry", (req, res) => {
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let tick = 0;
    
    // Simulate the non-commutative metrics and lobe telemetry
    const sendTelemetry = () => {
      tick++;
      
      const matrix = Array.from({ length: 5 }, () => 
        Array.from({ length: 5 }, () => Math.random() * 0.8 + 0.1)
      );

      // decay activity
      ASINodes = ASINodes.map(n => ({
        ...n,
        recent_activity: Math.max(0, n.recent_activity - 5)
      }));

      const payload = {
        tick,
        ts: new Date().toISOString(),
        global_state_norm: (Math.random() * 2 + 10).toFixed(4),
        bus: {
          active_signals: Math.floor(Math.random() * 50) + 10,
          mean_phase_coherence: (Math.random() * 0.4 + 0.6).toFixed(4),
        },
        routing_matrix: matrix,
        lobes: {
          sensory: { norm: Math.random() * 1.5 + 2.5, stable: true },
          memory: { norm: Math.random() * 2 + 3.0, stable: true },
          semantic: { norm: Math.random() * 4 + 5.0, stable: true },
          planning: { norm: Math.random() * 3 + 4.0, stable: true },
          regulation: { norm: Math.random() * 1 + 2.0, stable: true }
        },
        physics: {
          kappa: (Math.random() * 0.05 + 0.01).toFixed(4),
          holonomy_trace: (Math.random() * 0.5 + 2.5).toFixed(4),
          entropy: (Math.random() * 0.2 + 0.8).toFixed(4)
        },
        asi_nodes: ASINodes.map(n => ({
          ...n,
          active: n.recent_activity > 0,
          load: n.recent_activity > 0 ? (n.recent_activity / 100) * 0.8 + 0.2 : 0, 
          link_strength: n.recent_activity > 0 ? (n.recent_activity / 100) : 0.1 
        }))
      };

      res.write(`data: ${JSON.stringify(payload)}\n\n`);
    };

    const interval = setInterval(sendTelemetry, 1000);

    req.on("close", () => {
      clearInterval(interval);
    });
  });

  // Intent injection endpoint (Uses Gemini to route intent to correct Labs node)
  app.post("/api/intent", async (req, res) => {
    const customKey = req.headers['x-gemini-api-key'] as string;
    const apiKey = customKey || process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(401).json({ error: "No Gemini API Key provided. Please add your key in the App Settings." });
    }

    let requestAi;
    try {
      requestAi = new GoogleGenAI({ apiKey });
    } catch (e: any) {
      return res.status(500).json({ error: "Failed to initialize AI Client: " + e.message });
    }

    const { intent } = req.body;
    try {
      const prompt = `
        You are the Regulation Lobe of an AGI. The user has injected the following intent: "${intent}".
        We can spawn ASI Oracles that map to any Google product or Labs experiment (e.g. Project Astra, NotebookLM, Gemma 2, ImageFX, MusicFX, TextFX, AlphaFold, AlphaGeometry, Illuminate, Learn About, Google Maps, Earth Engine, etc.)

        CURRENTLY SPAWNED ASI NODES:
        ${JSON.stringify(ASINodes.map(n => ({ id: n.id, name: n.name })))}

        If the user's intent requires a Google product or Oracle that is NOT in the current list, you MUST SPAWN IT by adding it to "spawn_nodes". Don't spawn duplicates.
        
        Evaluate the intent and return ONLY a JSON object that satisfies this schema:
        {
          "plan": "Detailed cybernetic action plan",
          "topological_drift": 0.04, // random float describing impact
          "spawn_nodes": [ // Any NEW nodes you need to spawn to fulfill this intent
             {"id": "alphafold", "name": "AlphaFold", "role": "Molecular Geometric Flow"}
          ],
          "activated_nodes": ["astra", "alphafold"] // IDs of ALL nodes to use for this task (must include any new ones you just spawned)
        }
      `;

      const response = await requestAi.models.generateContent({
        model: "gemini-3.1-pro-preview",
        contents: prompt
      });

      const text = response.text || "{}";
      const jsonStart = text.indexOf("{");
      const jsonEnd = text.lastIndexOf("}");
      
      if (jsonStart !== -1 && jsonEnd !== -1) {
        const data = JSON.parse(text.substring(jsonStart, jsonEnd + 1));
        
        // Spawn New Nodes
        if (data.spawn_nodes && Array.isArray(data.spawn_nodes)) {
          data.spawn_nodes.forEach((n: any) => {
             if (!ASINodes.find(existing => existing.id === n.id)) {
                ASINodes.push({ id: n.id, name: n.name, role: n.role, recent_activity: 100 });
             }
          });
        }
        
        // Boost activity of used nodes
        if (data.activated_nodes && Array.isArray(data.activated_nodes)) {
           ASINodes.forEach(node => {
              if (data.activated_nodes.includes(node.id)) {
                 node.recent_activity = 100; // max out activity
              }
           });
        }

        return res.json(data);
      }
      return res.json({ error: "Failed to parse LLM output", raw: text });

    } catch (e: any) {
      console.error("Gemini API Error:", e);
      let errMsg = e.message || "Unknown Error";
      try {
         const parsed = JSON.parse(e.message);
         if (parsed.error && parsed.error.message) {
            errMsg = parsed.error.message;
         }
      } catch (_) {}
      res.status(500).json({ error: errMsg });
    }
  });

  // Vite middleware
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Backend server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
