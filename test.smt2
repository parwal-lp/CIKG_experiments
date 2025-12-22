;; −−− RELU DEFINITION −−−
(define-fun max ((x Real) (y Real)) Real (ite (< x y) y x))

;; −−− INPUT VARIABLES −−−
(define-fun X_0 () Real)
(define-fun X_1 () Real)

(check-sat)
(get-model)