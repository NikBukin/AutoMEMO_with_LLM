"use client";

import React from "react";
import { ChunkScore, Message, DocumentScore } from "@/app/types"; // Добавьте DocumentScore
import ReactMarkdown from "react-markdown";
import { FaDatabase } from "react-icons/fa";
import { BiError } from "react-icons/bi";
import { IoNewspaper } from "react-icons/io5";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { IoDocumentAttach } from "react-icons/io5";
import {
  oneDark,
  oneLight,
} from "react-syntax-highlighter/dist/cjs/styles/prism";

import VerbaButton from "../Navigation/VerbaButton";

import { Theme } from "@/app/types";

interface MEMOMessageProps {
  message: Message;
  message_index: number;
  selectedTheme: Theme;
  selectedDocument: string | null;
  setSelectedDocument: (s: string | null) => void;
  setSelectedDocumentScore: (s: string | null) => void;
  setSelectedChunkScore: (s: ChunkScore[]) => void;
}

const MEMOMessage: React.FC<MEMOMessageProps> = ({
  message,
  selectedTheme,
  selectedDocument,
  setSelectedDocument,
  message_index,
  setSelectedDocumentScore,
  setSelectedChunkScore,
}) => {
  const colorTable = {
    user: "bg-bg-verba",
    system: "bg-bg-alt-verba", // Цвет фона для системных сообщений
    loading: "bg-bg-alt-verba",
    error: "bg-warning-verba",
    retrieval: "bg-some-appropriate-color",
  };

  const messageBgClass = colorTable[message.type];

  const handleDocumentClick = (document: DocumentScore) => {
    setSelectedDocument(document.uuid);
    setSelectedDocumentScore(document.uuid); // Если это нужно
    setSelectedChunkScore(document.chunks);
  };

  return (
    <div
      className={`chat ${
        message.type === "user" ? "chat-end" : "chat-start"
      } w-full`}
    >
      <div className={`chat-bubble w-full ${messageBgClass}`}>
        {message.type === "error" && (
          <div className="flex flex-row gap-2 items-center text-text-verba">
            <BiError size={20} />
            <p>Error</p>
          </div>
        )}

        {/* Условный рендеринг для content */}
        {typeof message.content === "string" ? (
          <ReactMarkdown
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || "");
                return !inline && match ? (
                  <SyntaxHighlighter
                    {...props}
                    style={selectedTheme.theme === "dark" ? oneDark : oneLight}
                    language={match[1]}
                    PreTag="div"
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                ) : (
                  <code {...props} className={className}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        ) : (
          // Этот блок выполняется, если message.content является DocumentScore[]
          <div className="flex flex-col gap-2 p-2 w-full max-w-sm overflow-hidden">
            <h3 className="text-sm font-semibold text-text-verba">
              Найденные документы:
            </h3>
            {message.content.map((document: DocumentScore) => (
              <button
                key={document.uuid}
                className={`btn btn-sm flex flex-col justify-between items-center text-left max-w-full
                  ${
                    selectedDocument === document.uuid
                      ? "bg-secondary-verba hover:bg-secondary-verba"
                      : "bg-button-verba hover:bg-button-hover-verba text-text-verba"
                  }
                `}
                onClick={() => handleDocumentClick(document)}
              >
                <div className="flex flex-row items-center justify-between w-full">
                  <p
                    className="text-xs flex-grow truncate mr-2"
                    title={document.title}
                  >
                    {document.title}
                  </p>
                  <div className="flex gap-1 items-center text-text-verba flex-shrink-0">
                    <IoNewspaper size={12} />
                    <p className="text-sm">{document.chunks.length}</p>
                  </div>
                </div>
              </button>
            ))}
            {/* Кнопка "Прикрепить документ" - если это актуально */}
            <VerbaButton
              Icon={IoDocumentAttach}
              className="btn-sm btn-square"
              onClick={() =>
                (
                  document.getElementById(
                    "context-modal-" + message_index
                  ) as HTMLDialogElement
                ).showModal()
              }
            />
            <dialog id={"context-modal-" + message_index} className="modal">
              <div className="modal-box">
                <h3 className="font-bold text-lg">Context</h3>
                <p className="py-4">{message.context}</p>
                <div className="modal-action">
                  <form method="dialog">
                    <button className="btn focus:outline-none text-text-alt-verba bg-button-verba hover:bg-button-hover-verba hover:text-text-verba border-none shadow-none">
                      <p>Close</p>
                    </button>
                  </form>
                </div>
              </div>
            </dialog>
          </div>
        )}
      </div>
    </div>
  );
};

export default MEMOMessage;
