"use client";

import React, { useState, useEffect, useRef } from "react";
import { MdCancel, MdOutlineRefresh } from "react-icons/md";
import { TbPlugConnected } from "react-icons/tb";
import { IoIosSend } from "react-icons/io";
import VerbaButton from "../Navigation/VerbaButton";

import {
  updateRAGConfig,
  fetchDatacount,
  fetchRAGConfig,
  fetchSuggestions,
  fetchLabels,
  sendMemoRequest, // Import the new memo API call
} from "@/app/api";
import {
  Credentials,
  Suggestion,
  DataCountPayload,
  ChunkScore,
  Message,
  LabelsResponse,
  RAGConfig,
  Theme,
  DocumentFilter,
  MemoPayload, // Import MemoPayload type
  MemoResponse, // Import MemoResponse type
} from "@/app/types";

import InfoComponent from "../Navigation/InfoComponent";
import MEMOMessage from "./MEMOMessage"; // Assuming MEMOMessage exists

interface MEMOInterfaceProps {
  credentials: Credentials;
  setSelectedDocument: (s: string | null) => void;
  setSelectedChunkScore: (c: ChunkScore[]) => void;
  currentPage: string;
  selectedTheme: Theme;
  addStatusMessage: (
    message: string,
    type: "INFO" | "WARNING" | "SUCCESS" | "ERROR"
  ) => void;
  production: "Local" | "Demo" | "Production";
  documentFilter: DocumentFilter[];
  setDocumentFilter: React.Dispatch<React.SetStateAction<DocumentFilter[]>>;
  selectedDocument: string | null; // Receive selectedDocument from MEMOView
}

const MEMOInterface: React.FC<MEMOInterfaceProps> = ({
  credentials,
  setSelectedDocument,
  setSelectedChunkScore,
  currentPage,
  selectedTheme,
  addStatusMessage,
  production,
  documentFilter,
  setDocumentFilter,
  selectedDocument, // Use selectedDocument prop
}) => {
  const [messages, setMessages] = useState<Message[]>([
    { type: "system", content: selectedTheme.intro_message.text },
  ]);
  const [userInput, setUserInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [dataCount, setDataCount] = useState<DataCountPayload>({
    datacount: 0,
  });
  const [connected, setConnected] = useState<boolean>(true);
  const [currentSuggestions, setCurrentSuggestions] = useState<Suggestion[]>(
    []
  );
  const [selectedDocumentScore, setSelectedDocumentScore] = useState<
    string | null
  >(null);
  const [showConfig, setShowConfig] = useState<boolean>(false);
  const [labels, setLabels] = useState<LabelsResponse | null>(null);

  // New state for memo generation
  const [memoTemplate, setMemoTemplate] = useState<string>("");
  const [memoResult, setMemoResult] = useState<string>("");
  const [isGeneratingMemo, setIsGeneratingMemo] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMemo = async () => {
    if ((!selectedDocument && !userInput) || !memoTemplate) {
      addStatusMessage(
        "Please select a document or enter text, and provide a memo template.",
        "WARNING"
      );
      return;
    }

    setIsGeneratingMemo(true);
    addStatusMessage("Generating memo...", "INFO");

    const payload: MemoPayload = {
      memo_template: userInput,
      document_uuid: selectedDocument || undefined,
      credentials: credentials,
    };

    if (selectedDocument) {
      payload.document_uuid = selectedDocument;
    } else if (userInput) {
      payload.text_content = userInput;
    }

    try {
      const response: MemoResponse = await sendMemoRequest(payload);
      if (response && response.memo_text) {
        setMemoResult(response.memo_text);
        addStatusMessage("Memo generated successfully!", "SUCCESS");
      } else {
        setMemoResult("Error: Failed to generate memo.");
        addStatusMessage("Failed to generate memo.", "ERROR");
      }
    } catch (error) {
      console.error("Error generating memo:", error);
      setMemoResult(`Error: ${error instanceof Error ? error.message : String(error)}`);
      addStatusMessage(
        `Error generating memo: ${error instanceof Error ? error.message : String(error)}`,
        "ERROR"
      );
    } finally {
      setIsGeneratingMemo(false);
    }
  };

  const reconnectToVerba = async () => {
    // Reconnection logic (similar to ChatInterface)
    setConnected(false);
    addStatusMessage("Attempting to reconnect...", "INFO");
    try {
      const config = await fetchRAGConfig(credentials);
      if (config) {
        addStatusMessage("Reconnected successfully!", "SUCCESS");
        setConnected(true);
      } else {
        addStatusMessage("Failed to reconnect to Verba. Please check server.", "ERROR");
      }
    } catch (error) {
      addStatusMessage("Failed to reconnect to Verba. Please check server.", "ERROR");
      console.error("Reconnection error:", error);
    }
  };

  const handleDocumentSelect = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDocument(event.target.value === "" ? null : event.target.value);
  };

  const handleReset = () => {
    setSelectedDocument(null);
    setSelectedChunkScore([]);
    setUserInput("");
    setMemoTemplate("");
    setMemoResult("");
    setSelectedDocumentScore(null);
    setCurrentSuggestions([]);
    setMessages([{ type: "system", content: selectedTheme.intro_message.text }]);
  };


  return (
    <div className="flex flex-col h-full w-full justify-between items-center bg-bg-alt-verba rounded-lg">
      <div className="flex flex-row justify-between items-center w-full p-4">
        <div className="flex items-center gap-2">
          <img
            src={selectedTheme.image.src}
            alt="Theme Logo"
            className="w-8 h-8 rounded-full"
          />
          <h3 className="text-xl font-bold text-text-verba">
            {selectedTheme.title.text} - MEMO
          </h3>
        </div>
        <VerbaButton
          Icon={MdOutlineRefresh}
          onClick={handleReset}
          disabled={false}
          selected_color="bg-primary-verba"
          title="Reset MEMO"
        />
      </div>

      <div className="flex-grow w-full overflow-y-auto p-4 flex flex-col justify-start gap-4">
        <div className="flex flex-col gap-2">
          <label className="text-text-verba text-sm">
            Select Document for Memo (Optional):
          </label>
          <select
            className="select select-bordered w-full bg-bg-verba text-text-verba border-border-verba"
            value={selectedDocument || ""}
            onChange={handleDocumentSelect}
          >
            <option value="">No Document Selected (Enter text below)</option>
            {documentFilter.map((doc) => (
              <option key={doc.uuid} value={doc.uuid}>
                {doc.title}
              </option>
            ))}
          </select>
        </div>

        {!selectedDocument && (
          <div className="flex flex-col gap-2">
            <label className="text-text-verba text-sm">
              Enter Text Content for Memo (If no document selected):
            </label>
            <textarea
              className="textarea textarea-bordered w-full bg-bg-verba text-text-verba border-border-verba"
              placeholder="Enter text here..."
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              rows={6}
            ></textarea>
          </div>
        )}

        <div className="flex flex-col gap-2">
          <label className="text-text-verba text-sm">Memo Template:</label>
          <textarea
            className="textarea textarea-bordered w-full bg-bg-verba text-text-verba border-border-verba"
            placeholder="Enter your memo template. Use {context} to inject document content. Example: 'Write a summary of {context} in 100 words.'"
            value={memoTemplate}
            onChange={(e) => setMemoTemplate(e.target.value)}
            rows={6}
          ></textarea>
        </div>

        <VerbaButton
          type="button"
          Icon={IoIosSend}
          onClick={sendMemo}
          disabled={false}
          selected_color="bg-primary-verba"
          title="Отправить"
        />

        {memoResult && (
          <div className="flex flex-col gap-2">
            <h3 className="text-lg font-bold text-text-verba">Generated Memo:</h3>
            <div className="p-4 rounded-lg bg-bg-verba text-text-verba whitespace-pre-wrap">
              {memoResult}
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col p-4 w-full">
        {connected ? (
          <div className="flex gap-2 items-center justify-end w-full">
            <div className="flex gap-2">
              {/* This section can be repurposed if needed, or removed if no user input chat is desired */}
            </div>
          </div>
        ) : (
          <div className="flex gap-2 items-center justify-end w-full">
            <button
              onClick={reconnectToVerba}
              className="flex btn border-none text-text-verba bg-button-verba hover:bg-button-hover-verba gap-2 items-center"
            >
              <TbPlugConnected size={15} />
              <p>Reconnecting...</p>
              <span className="loading loading-spinner loading-xs"></span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default MEMOInterface;
