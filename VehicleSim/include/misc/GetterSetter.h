#ifndef VEHICLESIM_GETTERSETTER
#define VEHICLESIM_GETTERSETTER

namespace vlr
{
	template <typename baseClass, typename pointerType>
	class GetterSetter
	{
	private:
		enum Mode;

	public:
		typedef pointerType (baseClass::*getterPointer)() const;
		typedef const pointerType& (baseClass::*getterPointerConstRef)() const;
		typedef void (baseClass::*setterPointer)(pointerType);
		typedef void (baseClass::*setterPointerConstRef)(const pointerType&);

		GetterSetter(baseClass* base, pointerType* pointer)
			: _mode(POINTER), _pointer(pointer)
		{

		}

		GetterSetter(baseClass* base, getterPointer getterFunc,
			setterPointer setterFunc)
			: _mode(FUNC), _getterFunc(getterFunc), _setterFunc(setterFunc)
		{

		}

		GetterSetter(baseClass* base, getterPointer getterFunc,
			setterPointerConstRef setterFuncConstRef)
			: _mode(FUNC_SETCONSTREF), _getterFunc(getterFunc),
			_setterFuncConstRef(setterFuncConstRef)
		{

		}

		GetterSetter(baseClass* base, getterPointerConstRef getterFunc,
			setterPointerConstRef setterFuncConstRef)
			: _mode(FUNC_CONSTREF), _getterFuncConstRef(getterFunc),
			_setterFuncConstRef(setterFuncConstRef)
		{

		}

		pointerType getValue()
		{
			switch (_mode)
			{
			case POINTER:
				return *_pointer;
				break;
			case FUNC:
			case FUNC_SETCONSTREF:
				return (_base->*_getterFunc)();
				break;
			case FUNC_CONSTREF:
				return (_base->*_getterFuncConstRef)();
				break;
			default:
				return pointerType();
				break;
			}
		}

		void setBase(baseClass* base)
		{
			_base = base;
		}

		void setValue(pointerType val)
		{
			switch (_mode)
			{
			case POINTER:
				*_pointer = val;
				break;
			case FUNC:
				if (_setterFunc != nullptr)
					(_base->*_setterFunc)(val);
				break;
			case FUNC_CONSTREF:
			case FUNC_SETCONSTREF:
				if (_setterFuncConstRef != nullptr)
					(_base->*_setterFuncConstRef)(val);
				break;
			default:
				break;
			}
		}
		
	private:
		enum Mode
		{
			POINTER,
			FUNC,
			FUNC_SETCONSTREF,
			FUNC_CONSTREF
		};

		Mode _mode;

		baseClass* _base;
		pointerType* _pointer;
		getterPointer _getterFunc;
		getterPointerConstRef _getterFuncConstRef;
		setterPointer _setterFunc;
		setterPointerConstRef _setterFuncConstRef;
	};
}

#endif /* VEHICLESIM_GETTERSETTER */
