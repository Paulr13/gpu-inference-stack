;; llama-server shepherd user service — generic template (Guix Home).
;; Usage: include this service in your guix-home config's services list,
;; adjust the paths, then: guix home reconfigure
;; Needs: (use-module (gnu home services shepherd)) (use-module (guix gexp))

(simple-service 'llama-server-shepherd
                home-shepherd-service-type
                (list (shepherd-service
                       (documentation "llama.cpp OpenAI-compatible server on 127.0.0.1:8080")
                       (provision '(llama-server))
                       (requirement '())
                       (respawn? #t)
                       (start #~(let ((home (getenv "HOME")))
                                  (make-forkexec-constructor
                                   (list (string-append home "/llama/bin/llama-server")
                                         "--model" (string-append home "/llama/models/model.gguf")
                                         "--host" "127.0.0.1"
                                         "--port" "8080"
                                         "--ctx-size" "32768")
                                   #:log-file (string-append home "/llama/llama-server.log"))))
                       (stop #~(make-kill-destructor)))))